"""
``AlchemiModel``: an nvalchemi ``BaseModelMixin`` proxy whose real model
runs in a Rootstock worker subprocess.

The proxy advertises the worker wrapper's ``ModelConfig`` with one
deliberate rewrite: ``autograd_outputs`` is emptied. Autograd graphs
cannot cross a process boundary, so the proxy presents forces/stress as
analytical (direct) outputs — the worker still computes them via the
model's own autograd, and the values are bitwise the same physics. The
costs of the rewrite: the proxy cannot join a shared-autograd pipeline
group, and ``gradient_keys`` requests are unsupported.

Requires nvalchemi in the *calling* environment (it hosts the engine);
the model family's stack lives only in the worker env.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from torch import nn

from rootstock.batched.wire import arrays_to_tensors, recv_msg, send_msg, tensors_to_arrays
from rootstock.clusters import get_cluster
from rootstock.config import resolve_default_root
from rootstock.environment import find_batched_env_for_checkpoint
from rootstock.exceptions import RootstockError
from rootstock.layout import ensure_layout_compatible, resolve_cache_root
from rootstock.spawn import spawn_in_env

logger = logging.getLogger("rootstock.batched.model")

BATCHED_WORKER_WRAPPER = """\
import json, os, sys

with open(sys.argv[1]) as f:
    spec = json.load(f)

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
try:
    import prewarm
    prewarm.prewarm_from_spec(spec)
except ImportError:
    pass  # wrapper run without staged helper; warming is optional
finally:
    sys.path.remove(_here)

sys.path.insert(0, spec["env_dir"])
from rootstock.batched.worker import run_batched_worker
from env_source import setup_batched

run_batched_worker(
    setup_fn=setup_batched,
    checkpoint=spec["checkpoint"],
    device=spec["device"],
    socket_path=spec["socket_path"],
    setup_kwargs=spec["setup_kwargs"],
)
"""


class BatchedWorkerError(RootstockError, RuntimeError):
    """The batched worker died or reported a compute error."""


class _CudaTransportError(BatchedWorkerError):
    """CUDA IPC setup failed with the socket stream still aligned.

    Raised only at points where no request is in flight (client-side export
    failure, or a worker error reply that was fully consumed), so the auto
    transport can retry the same batch over the socket transport.
    """


def _tail(text: str, limit: int = 8192) -> str:
    return text[-limit:] if len(text) > limit else text


class AlchemiModel(nn.Module, BaseModelMixin):
    """Batched nvalchemi model served from an isolated Rootstock environment.

    The checkpoint id is the same canonical id the ASE path uses — same
    weights, same resolution: the hosting env is found by walking the
    installed envs for one that declares the id *and* offers
    ``setup_batched``.

    Args:
        checkpoint: Canonical checkpoint id (e.g. ``"uma-s-1p1"``).
        cluster: Known cluster name; mutually exclusive with ``root``.
        root: Rootstock install root. When neither is given, ROOTSTOCK_ROOT
            and then the configured default apply, as for the calculator.
        cache_root: Optional cache override; the install's own declaration
            decides otherwise.
        device: Worker-side compute device (``"cuda"`` / ``"cpu"``).
        setup_kwargs: Extra kwargs for ``setup_batched`` (e.g. UMA ``task``).
        neighbor_mode: ``"worker"`` (default) — the worker builds neighbor
            lists next to the model and the proxy advertises no
            ``neighbor_config``; ``"engine"`` — the proxy keeps the real
            ``neighbor_config`` so the engine's NeighborListHook builds the
            list, which then ships over the wire each step.
        transport: ``"auto"`` (default) picks per install: CUDA IPC when the
            worker computes on CUDA (shared GPU buffers, ~100-byte control
            messages), falling back to the raw-bytes socket transport if
            registration fails; explicit ``"socket"`` / ``"cuda"`` pin one,
            mainly for benchmarking.
        accept_timeout: Seconds to wait for the worker to load the model
            and connect.
        collect_stats: Record per-call timing/bytes in ``self.stats``.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        cluster: str | None = None,
        root: str | Path | None = None,
        cache_root: str | Path | None = None,
        device: str = "cuda",
        setup_kwargs: dict | None = None,
        neighbor_mode: str = "worker",
        transport: str = "auto",
        accept_timeout: float = 1800.0,
        collect_stats: bool = True,
    ) -> None:
        super().__init__()
        if neighbor_mode not in ("worker", "engine"):
            raise ValueError(f"neighbor_mode must be 'worker' or 'engine', got {neighbor_mode!r}")
        if transport not in ("auto", "socket", "cuda"):
            raise ValueError(f"transport must be 'auto', 'socket' or 'cuda', got {transport!r}")
        if transport == "cuda" and device == "cpu":
            raise ValueError("transport='cuda' requires a CUDA worker device")
        if cluster is not None and root is not None:
            raise ValueError("Cannot specify both 'cluster' and 'root'")
        if cluster is not None:
            root = get_cluster(cluster).root
        elif root is not None:
            root = Path(root)
        else:
            root = resolve_default_root()
            if root is None:
                raise ValueError(
                    "Must specify 'cluster' or 'root' (or set the ROOTSTOCK_ROOT "
                    "environment variable, or configure root in "
                    "~/.config/rootstock/config.toml)"
                )
        ensure_layout_compatible(root)

        self.checkpoint = checkpoint
        self.worker_device = device
        self.neighbor_mode = neighbor_mode
        self._transport = "cuda" if transport == "auto" and device != "cpu" else transport
        if self._transport == "auto":
            self._transport = "socket"
        self._transport_pinned = transport != "auto"
        self.collect_stats = collect_stats
        self.stats: list[dict] = []
        self._ipc_sig: tuple | None = None
        self._ipc_bufs: dict[str, torch.Tensor] = {}
        self._ipc_outs: dict[str, torch.Tensor] = {}

        env = find_batched_env_for_checkpoint(root, checkpoint, cluster)
        self._exit_stack = contextlib.ExitStack()
        try:
            self._start_worker(
                root,
                env,
                setup_kwargs or {},
                accept_timeout,
                cache_root=resolve_cache_root(root, explicit=cache_root),
            )
            self._handshake(accept_timeout)
        except BaseException:
            self._exit_stack.close()
            raise

    @property
    def transport(self) -> str:
        """The transport in effect (``"cuda"`` or ``"socket"``)."""
        return self._transport

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_worker(
        self,
        root: Path,
        env: str,
        setup_kwargs: dict,
        accept_timeout: float,
        cache_root: Path | None = None,
    ):
        import subprocess

        self._socket_dir = tempfile.mkdtemp(prefix="rootstock_batched_")
        self._exit_stack.callback(self._rm_socket_dir)
        socket_path = os.path.join(self._socket_dir, "worker.sock")

        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._exit_stack.callback(self._listener.close)
        self._listener.bind(socket_path)
        self._listener.listen(1)
        self._listener.settimeout(accept_timeout)

        payload = {
            "checkpoint": self.checkpoint,
            "device": self.worker_device,
            "socket_path": socket_path,
            "setup_kwargs": setup_kwargs,
        }
        spec = self._exit_stack.enter_context(
            spawn_in_env(root, env, BATCHED_WORKER_WRAPPER, payload, cache_root=cache_root)
        )
        # Files, not pipes: a noisy model load can exceed the OS pipe buffer
        # and deadlock before the worker ever connects.
        self._stdout_file = tempfile.TemporaryFile()
        self._stderr_file = tempfile.TemporaryFile()
        self._exit_stack.callback(self._stdout_file.close)
        self._exit_stack.callback(self._stderr_file.close)
        logger.debug("spawning batched worker: %s", " ".join(spec.cmd))
        self._process = subprocess.Popen(
            spec.cmd,
            env=spec.env,
            cwd=spec.cwd,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )
        self._exit_stack.callback(self._terminate_process)

    def _rm_socket_dir(self):
        import shutil

        shutil.rmtree(self._socket_dir, ignore_errors=True)

    def _terminate_process(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()

    def _worker_output(self) -> str:
        out = []
        for name, f in (("stdout", self._stdout_file), ("stderr", self._stderr_file)):
            try:
                f.seek(0)
                text = f.read().decode(errors="replace").strip()
            except Exception:
                text = ""
            if text:
                out.append(f"--- worker {name} ---\n{_tail(text)}")
        return "\n".join(out)

    def _fail(self, context: str) -> BatchedWorkerError:
        return BatchedWorkerError(f"{context}\n{self._worker_output()}")

    def _handshake(self, accept_timeout: float):
        try:
            self._sock, _ = self._listener.accept()
        except TimeoutError:
            raise self._fail(f"batched worker did not connect within {accept_timeout}s") from None
        self._exit_stack.callback(self._sock.close)
        self._sock.settimeout(accept_timeout)

        header, _, _ = recv_msg(self._sock)
        if header.get("type") != "hello":
            raise self._fail(f"expected hello, got {header.get('type')!r}")

        self._worker_info = header
        worker_cfg = ModelConfig.model_validate_json(header["model_config"])
        self._worker_neighbor_config = worker_cfg.neighbor_config
        self._direct_keys = set(header.get("direct_derivative_keys", []))
        self._embedding_shapes = {
            k: tuple(v) for k, v in header.get("embedding_shapes", {}).items()
        }
        self._wire_keys = list(header["wire_keys"])
        self._optional_wire_keys = list(header.get("optional_wire_keys", []))

        # The autograd rewrite: values computed via autograd worker-side
        # arrive here as plain data, so the proxy declares them direct.
        self.model_config = ModelConfig(
            outputs=worker_cfg.outputs,
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            required_inputs=worker_cfg.required_inputs,
            optional_inputs=worker_cfg.optional_inputs,
            supports_pbc=worker_cfg.supports_pbc,
            needs_pbc=worker_cfg.needs_pbc,
            neighbor_config=(
                worker_cfg.neighbor_config if self.neighbor_mode == "engine" else None
            ),
            active_outputs=set(worker_cfg.active_outputs),
        )
        logger.info(
            "batched worker ready: %s (load %.1fs, worker torch %s)",
            self.checkpoint,
            header.get("load_seconds", float("nan")),
            header.get("torch_version", "?"),
        )

    def clear_worker_cache(self):
        """Release the worker's cached CUDA allocator blocks back to the driver."""
        send_msg(self._sock, {"type": "clear_cache"}, {})
        reply, _, _ = recv_msg(self._sock)
        if reply.get("type") != "result":
            raise self._fail(f"expected result, got {reply.get('type')!r}")

    def close(self):
        """Shut the worker down; the proxy is unusable afterwards."""
        with contextlib.suppress(Exception):
            send_msg(self._sock, {"type": "shutdown"}, {})
            self._sock.settimeout(10)
            recv_msg(self._sock)
        self._exit_stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # BaseModelMixin surface
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return self._embedding_shapes

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError("RootstockModel does not proxy embeddings yet")

    def direct_derivative_keys(self) -> set[str]:
        return set(self._direct_keys)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _gather(self, data) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for key in self._wire_keys:
            if key == "num_nodes_per_graph":
                tensors[key] = data.num_nodes_per_graph.to(torch.long)
                continue
            value = getattr(data, key, None)
            if value is None:
                raise BatchedWorkerError(f"batch is missing required wire key {key!r}")
            tensors[key] = value
        for key in self._optional_wire_keys:
            value = getattr(data, key, None)
            if value is not None:
                tensors[key] = value
        if self.neighbor_mode == "engine":
            nl = getattr(data, "neighbor_list", None)
            if nl is None:
                raise BatchedWorkerError(
                    "neighbor_mode='engine' but the batch has no neighbor_list; "
                    "register the model's NeighborListHook (make_neighbor_hooks())"
                )
            tensors["neighbor_list"] = nl
            shifts = getattr(data, "neighbor_list_shifts", None)
            if shifts is not None:
                tensors["neighbor_list_shifts"] = shifts
        return tensors

    def _forward_cuda(self, data) -> OrderedDict:
        from rootstock.batched.cuda_ipc import export_cuda_tensor, import_cuda_tensor

        device = data.positions.device
        if device.type != "cuda":
            raise BatchedWorkerError("transport='cuda' requires the batch on a CUDA device")
        stat: dict[str, Any] = {"transport": "cuda"}
        active = sorted(self.model_config.active_outputs)

        t0 = time.perf_counter()
        tensors = self._gather(data)
        sig = (
            tuple((k, str(t.dtype), tuple(t.shape)) for k, t in sorted(tensors.items())),
            tuple(active),
        )
        if sig != self._ipc_sig:
            try:
                self._ipc_bufs = {k: t.detach().contiguous().clone() for k, t in tensors.items()}
                descs = {k: export_cuda_tensor(t) for k, t in self._ipc_bufs.items()}
            except Exception as exc:
                raise _CudaTransportError(f"CUDA IPC export failed: {exc}") from exc
            torch.cuda.synchronize(device)
            send_msg(
                self._sock,
                {"type": "register_inputs", "descriptors": descs, "active_outputs": active},
                {},
            )
            reply, _, _ = recv_msg(self._sock)
            if reply.get("type") == "error":
                raise _CudaTransportError(
                    f"worker registration failed:\n{reply.get('traceback', '')}"
                )
            if reply.get("type") != "registered":
                raise self._fail(f"expected registered, got {reply.get('type')!r}")
            try:
                self._ipc_outs = {k: import_cuda_tensor(d) for k, d in reply["descriptors"].items()}
            except Exception as exc:
                raise _CudaTransportError(f"CUDA IPC import failed: {exc}") from exc
            self._ipc_sig = sig
        else:
            for key, t in tensors.items():
                self._ipc_bufs[key].copy_(t.detach())
        torch.cuda.synchronize(device)
        stat["gather_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        stat["bytes_sent"] = send_msg(
            self._sock, {"type": "compute_ipc", "active_outputs": active}, {}
        )
        reply, _, nbytes = recv_msg(self._sock)
        stat["bytes_received"] = nbytes
        stat["roundtrip_s"] = time.perf_counter() - t0

        if reply.get("type") == "error":
            raise BatchedWorkerError(f"worker compute failed:\n{reply.get('traceback', '')}")
        if reply.get("type") != "done":
            raise self._fail(f"expected done, got {reply.get('type')!r}")

        t0 = time.perf_counter()
        present = set(reply.get("present", []))
        out_tensors = {k: self._ipc_outs[k].clone() for k in present}
        torch.cuda.synchronize(device)
        stat["to_device_s"] = time.perf_counter() - t0
        stat["worker"] = reply.get("timings", {})
        if self.collect_stats:
            self.stats.append(stat)

        output = OrderedDict((key, None) for key in sorted(self.output_data()))
        for key in output:
            value = out_tensors.get(key)
            if value is not None:
                if key == "energy" and value.ndim == 1:
                    value = value.unsqueeze(-1)
                output[key] = value
        return output

    def forward(self, data, **kwargs) -> OrderedDict:
        if self._transport == "cuda":
            try:
                return self._forward_cuda(data)
            except _CudaTransportError:
                if self._transport_pinned:
                    raise
                logger.warning(
                    "CUDA IPC transport unavailable; falling back to the socket transport",
                    exc_info=True,
                )
                self._transport = "socket"
        return self._forward_socket(data)

    def _forward_socket(self, data) -> OrderedDict:
        device = data.positions.device
        stat: dict[str, Any] = {}

        t0 = time.perf_counter()
        tensors = self._gather(data)
        arrays, bf16 = tensors_to_arrays(tensors)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stat["gather_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        header = {
            "type": "compute",
            "active_outputs": sorted(self.model_config.active_outputs),
            "_bf16": bf16,
        }
        stat["bytes_sent"] = send_msg(self._sock, header, arrays)
        reply, out_arrays, nbytes = recv_msg(self._sock)
        stat["bytes_received"] = nbytes
        stat["roundtrip_s"] = time.perf_counter() - t0

        if reply.get("type") == "error":
            raise BatchedWorkerError(f"worker compute failed:\n{reply.get('traceback', '')}")
        if reply.get("type") != "result":
            raise self._fail(f"expected result, got {reply.get('type')!r}")

        t0 = time.perf_counter()
        out_tensors = arrays_to_tensors(reply, out_arrays, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stat["to_device_s"] = time.perf_counter() - t0
        stat["worker"] = reply.get("timings", {})
        if self.collect_stats:
            self.stats.append(stat)

        output = OrderedDict((key, None) for key in sorted(self.output_data()))
        for key in output:
            value = out_tensors.get(key)
            if value is not None:
                if key == "energy" and value.ndim == 1:
                    value = value.unsqueeze(-1)
                output[key] = value
        return output
