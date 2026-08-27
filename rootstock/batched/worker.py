"""
Batched worker: hosts a real nvalchemi model wrapper inside a pre-built
environment and serves ``compute`` requests over the wire protocol.

Runs with the env's Python (spawned via ``spawn_in_env`` with
``BATCHED_WORKER_WRAPPER``); the env carries nvalchemi plus the model
family's stack. The env source must define::

    def setup_batched(checkpoint: str, device: str = "cuda", **kwargs):
        return <BaseModelMixin wrapper>            # or (wrapper, options)

Options (all optional): ``{"compute_neighbors": bool}`` — whether the
worker builds the neighbor list before each forward when the client did
not ship one. Defaults to the wrapper's own declaration
(``model_config.needs_neighborlist``); env sources whose model builds its
graph internally (e.g. UMA's fairchem path) should pass ``False``.

Two transports serve compute requests:

- ``compute`` — tensors travel as raw bytes over the socket.
- ``register_inputs`` + ``compute_ipc`` — CUDA IPC: the client shares
  input buffers by memory handle; the worker maps them, runs one forward
  to learn output shapes, and shares its own output buffers back. Each
  step is then a GPU-side ``copy_`` plus a tiny control message.

The worker reconstructs a real ``nvalchemi.data.Batch`` from the wire
tensors: rebuilt via ``Batch.from_data_list`` only when the batch
segmentation (per-system atom counts) changes, tensor-updated in place
otherwise, so steady-state MD pays tensor copies, not Python batch
assembly. A client-shipped COO neighbor list is installed directly as
the batch's edges group (global atom indices, sentinel rows preserved),
mirroring ``nvalchemi.neighbors._write_neighbor_data_to_batch``.
"""

from __future__ import annotations

import logging
import socket
import time
import traceback

from rootstock.batched.wire import (
    PROTOCOL_VERSION,
    arrays_to_tensors,
    recv_msg,
    send_msg,
    tensors_to_arrays,
)

logger = logging.getLogger("rootstock.batched.worker")

# Wire keys that describe batch structure rather than model inputs.
_STRUCTURAL_KEYS = ("positions", "atomic_numbers", "num_nodes_per_graph")
# Neighbor keys, shipped only when the client runs the neighbor hook itself.
# Only the COO pair is supported on the wire; MATRIX-format models use
# worker-side neighbor construction.
_NEIGHBOR_KEYS = ("neighbor_list", "neighbor_list_shifts")
# Per-system keys, sliced ``[i : i + 1]`` when splitting into AtomicData.
_SYSTEM_KEYS = ("cell", "pbc", "charge", "spin", "mult", "tags")


def _wire_keys(model_config) -> tuple[list[str], list[str]]:
    """Derive (required, optional) wire keys from a wrapper's ModelConfig."""
    required = list(_STRUCTURAL_KEYS)
    optional = set(model_config.optional_inputs)
    optional.update(model_config.required_inputs)
    if model_config.supports_pbc:
        optional.update({"cell", "pbc"})
    optional.difference_update(_NEIGHBOR_KEYS)
    optional.difference_update(required)
    return required, sorted(optional)


class _BatchCache:
    """Owns the worker-side Batch; rebuilds only on segmentation change."""

    def __init__(self, device):
        self.device = device
        self.batch = None
        self._segmentation = None
        self._keys = None

    def update(self, tensors) -> object:
        """Refresh the cached Batch from wire tensors (neighbor keys excluded)."""
        import torch
        from nvalchemi.data import AtomicData, Batch

        counts = tensors["num_nodes_per_graph"].to(torch.long)
        segmentation = tuple(counts.tolist())
        fields = {k: v for k, v in tensors.items() if k != "num_nodes_per_graph"}
        field_keys = tuple(sorted(fields))

        if (
            self.batch is not None
            and segmentation == self._segmentation
            and field_keys == self._keys
        ):
            for key, value in fields.items():
                setattr(self.batch, key, value)
            return self.batch

        data_list = []
        node_start = 0
        for i, n in enumerate(segmentation):
            item = {}
            for key, value in fields.items():
                if key in _SYSTEM_KEYS:
                    item[key] = value[i : i + 1]
                else:
                    item[key] = value[node_start : node_start + n]
            data_list.append(AtomicData(**item))
            node_start += n
        self.batch = Batch.from_data_list(data_list, device=self.device)
        self._segmentation = segmentation
        self._keys = field_keys
        return self.batch


def _install_edges(batch, neighbor_list, shifts, cutoff) -> None:
    """Install a client-shipped COO neighbor list as the batch's edges group."""
    import torch
    from nvalchemi.data.level_storage import SegmentedLevelStorage

    src = neighbor_list[:, 0].long()
    graph_per_edge = batch.batch_idx[src]
    seg_lengths = torch.bincount(graph_per_edge, minlength=batch.num_graphs).to(torch.int32)
    data = {"neighbor_list": neighbor_list}
    if shifts is not None:
        data["neighbor_list_shifts"] = shifts
    batch._storage.groups["edges"] = SegmentedLevelStorage(
        data=data,
        device=batch.device,
        segment_lengths=seg_lengths,
        validate=False,
    )
    batch._neighbor_list_cutoff = cutoff


class _WorkerLoop:
    """One connected serving session."""

    def __init__(self, sock, wrapper, options, device):
        import torch

        self.torch = torch
        self.sock = sock
        self.wrapper = wrapper
        self.device = torch.device(device)
        self.cache = _BatchCache(self.device)
        cfg = wrapper.model_config
        self.neighbor_config = cfg.neighbor_config
        self.compute_nl = options.get("compute_neighbors", cfg.needs_neighborlist)
        self.ipc_inputs = None
        self.export_bufs = None

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _run_forward(self, tensors, active_outputs, timings):
        """Shared compute path: batch refresh -> neighbors -> forward."""
        t0 = time.perf_counter()
        tensors = dict(tensors)
        shipped_nl = tensors.pop("neighbor_list", None)
        shipped_shifts = tensors.pop("neighbor_list_shifts", None)
        batch = self.cache.update(tensors)
        if shipped_nl is not None:
            _install_edges(batch, shipped_nl, shipped_shifts, self.neighbor_config.cutoff)
        timings["deserialize_s"] += time.perf_counter() - t0

        if active_outputs:
            self.wrapper.model_config.active_outputs = set(active_outputs)

        t0 = time.perf_counter()
        if self.compute_nl and shipped_nl is None:
            from nvalchemi.neighbors import compute_neighbors

            compute_neighbors(batch, config=self.neighbor_config)
        timings["neighbors_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        outputs = self.wrapper(batch)
        self._sync()
        timings["forward_s"] = time.perf_counter() - t0
        return outputs

    # -- socket transport ------------------------------------------------

    def handle_compute(self, header, arrays):
        timings = {"deserialize_s": 0.0}
        t0 = time.perf_counter()
        tensors = arrays_to_tensors(header, arrays, self.device)
        timings["deserialize_s"] = time.perf_counter() - t0
        outputs = self._run_forward(tensors, header.get("active_outputs"), timings)

        t0 = time.perf_counter()
        out_tensors = {k: v for k, v in outputs.items() if v is not None}
        out_arrays, bf16 = tensors_to_arrays(out_tensors)
        timings["serialize_s"] = time.perf_counter() - t0
        send_msg(self.sock, {"type": "result", "timings": timings, "_bf16": bf16}, out_arrays)

    # -- CUDA IPC transport ----------------------------------------------

    def handle_register_inputs(self, header):
        from rootstock.batched.cuda_ipc import export_cuda_tensor, import_cuda_tensor

        self.ipc_inputs = {k: import_cuda_tensor(d) for k, d in header["descriptors"].items()}
        # One forward to learn output shapes, then share output buffers back.
        outputs = self._run_forward(
            self.ipc_inputs, header.get("active_outputs"), {"deserialize_s": 0.0}
        )
        self.export_bufs = {
            k: v.detach().contiguous().clone() for k, v in outputs.items() if v is not None
        }
        self._sync()
        out_descs = {k: export_cuda_tensor(t) for k, t in self.export_bufs.items()}
        send_msg(self.sock, {"type": "registered", "descriptors": out_descs}, {})

    def handle_compute_ipc(self, header):
        if self.ipc_inputs is None:
            raise RuntimeError("compute_ipc before register_inputs")
        timings = {"deserialize_s": 0.0}
        outputs = self._run_forward(self.ipc_inputs, header.get("active_outputs"), timings)

        t0 = time.perf_counter()
        present = []
        for key, value in outputs.items():
            if value is None:
                continue
            buf = self.export_bufs.get(key)
            if buf is None or buf.shape != value.shape:
                raise RuntimeError(
                    f"output {key!r} does not match the registered buffer; "
                    "re-register (active_outputs changed after registration?)"
                )
            buf.copy_(value.detach())
            present.append(key)
        self._sync()
        timings["serialize_s"] = time.perf_counter() - t0
        send_msg(self.sock, {"type": "done", "timings": timings, "present": present}, {})

    # -- dispatch --------------------------------------------------------

    def serve(self):
        while True:
            header, arrays, _ = recv_msg(self.sock)
            msg_type = header.get("type")
            if msg_type == "shutdown":
                send_msg(self.sock, {"type": "bye"}, {})
                return
            try:
                if msg_type == "compute":
                    self.handle_compute(header, arrays)
                elif msg_type == "register_inputs":
                    self.handle_register_inputs(header)
                elif msg_type == "compute_ipc":
                    self.handle_compute_ipc(header)
                elif msg_type == "clear_cache":
                    if self.device.type == "cuda":
                        self.torch.cuda.empty_cache()
                    send_msg(self.sock, {"type": "result", "timings": {}}, {})
                else:
                    send_msg(
                        self.sock,
                        {"type": "error", "traceback": f"unknown message {msg_type!r}"},
                        {},
                    )
            except Exception:
                logger.exception("%s failed", msg_type)
                send_msg(self.sock, {"type": "error", "traceback": traceback.format_exc()}, {})


def run_batched_worker(setup_fn, checkpoint, device, socket_path, setup_kwargs=None):
    """Load the wrapper via ``setup_fn`` and serve compute requests until shutdown."""
    logging.basicConfig(level=logging.INFO)
    import torch

    setup_kwargs = setup_kwargs or {}
    logger.info("loading %s on %s", checkpoint, device)
    t0 = time.perf_counter()
    result = setup_fn(checkpoint, device, **setup_kwargs)
    wrapper, options = result if isinstance(result, tuple) else (result, {})
    load_s = time.perf_counter() - t0
    logger.info("model loaded in %.1fs", load_s)

    cfg = wrapper.model_config
    required, optional = _wire_keys(cfg)
    try:
        embedding_shapes = {k: list(v) for k, v in wrapper.embedding_shapes.items()}
    except Exception:
        embedding_shapes = {}
    direct_keys = sorted(set(wrapper.direct_derivative_keys()) | set(cfg.autograd_outputs))

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    loop = _WorkerLoop(sock, wrapper, options, device)
    send_msg(
        sock,
        {
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
            "model_config": cfg.model_dump_json(),
            "wire_keys": required,
            "optional_wire_keys": optional,
            "direct_derivative_keys": direct_keys,
            "embedding_shapes": embedding_shapes,
            "compute_neighbors": loop.compute_nl,
            "load_seconds": load_s,
            "torch_version": torch.__version__,
        },
        {},
    )
    loop.serve()
    sock.close()
