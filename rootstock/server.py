"""
Socket server for Rootstock.

This runs in the main process and acts as an i-PI server,
sending atomic positions and receiving forces from a worker process.
"""

import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .exceptions import RootstockError
from .protocol import (
    IPIProtocol,
    SocketClosed,
    create_private_socket_path,
    create_server_socket,
)
from .usage import record_session

logger = logging.getLogger("rootstock.server")


class WorkerDiedError(RootstockError, RuntimeError):
    """The worker process failed at the socket level (died, hung, or the
    connection broke) — as opposed to reporting a calculation error in-band
    while staying healthy. The message carries the post-mortem: process fate
    plus captured output tails. A server that raised this cannot serve
    further calls; the calculator tears it down and starts fresh on the next
    calculation."""


def _tail(text: str, limit: int = 8192) -> str:
    """Last ``limit`` characters of ``text`` — worker output can be huge
    (chatty model loads), and the useful part of a crash is at the end."""
    if len(text) <= limit:
        return text
    return f"...[{len(text) - limit} chars truncated]...\n{text[-limit:]}"


def _worker_error_from_extra(extra: bytes) -> str | None:
    """Extract an in-band worker error from the FORCEREADY extra field.

    Workers (1.0+) report calculation failures as a JSON object
    {"error": "<traceback>"} in the otherwise-unused extra payload. Anything
    that isn't such an object — the b"\\x00" padding byte, empty bytes, or a
    future non-error use of the field — is not an error.
    """
    if not extra or not extra.startswith(b"{"):
        return None
    try:
        payload = json.loads(extra.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(payload, dict):
        return payload.get("error")
    return None


class RootstockServer:
    """
    Server that communicates with an MLIP worker process via i-PI protocol.

    The server:
    1. Creates a Unix domain socket
    2. Launches a worker subprocess using pre-built environment
    3. Accepts the worker's connection
    4. Sends positions, receives forces

    Example:
        with RootstockServer(
            env_name="mace",
            checkpoint="mace-mp-0-medium",
            device="cuda",
            root=Path("/vol/rootstock"),
        ) as server:
            energy, forces, virial = server.calculate(positions, cell, numbers)
    """

    def __init__(
        self,
        env_name: str,
        checkpoint: str,
        device: str = "cuda",
        socket_name: str = "rootstock",
        root: Path | None = None,
        cache_root: Path | None = None,
        timeout: float = 600.0,
        setup_kwargs: dict | None = None,
        checkpoint_path: str | None = None,
        usage_client: str | None = "calculator",
    ):
        """
        Initialize the server.

        Server-side diagnostics go to the ``rootstock.server`` logger
        (lifecycle at INFO, detail at DEBUG); the wire trace goes to
        ``rootstock.protocol`` at DEBUG. Enable with e.g.
        ``logging.basicConfig(level=logging.DEBUG)``.

        Args:
            env_name: Name of pre-built environment (e.g., "mace")
            checkpoint: Canonical checkpoint id passed to the env's setup()
            device: Device string to pass to setup()
            socket_name: Name for the Unix socket. The socket is created as
                ipi_<name> inside a fresh private (0700) temp directory on
                start(), so it is unreachable by other local users.
            root: Root directory for environments and cache (required)
            timeout: Socket timeout in seconds for worker operations.
                The default (600 s) matches what checkpoint verification
                uses, so the first real force call — which may pay for
                torch.compile or large neighbor lists — runs under the
                same envelope verification exercised.
            setup_kwargs: Extra keyword arguments forwarded to setup()
            checkpoint_path: Path to a local (user-registered) weights file.
                When set, the worker loads via the env's setup_from_path()
                hook instead of setup(); ``checkpoint`` is then only a label.
            usage_client: Client label written into the session's anonymous
                usage record (see usage.py), or None to record nothing —
                verification passes None so synthetic smoke-test sessions
                never pollute the spool.
        """
        if root is None:
            raise ValueError("root is required for pre-built environments")

        self.socket_name = socket_name
        # Created by start(): the socket lives in a private per-server temp
        # directory that stop() removes.
        self.socket_path: str | None = None
        self._socket_dir: str | None = None
        self.timeout = timeout

        self.env_name = env_name
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.root = Path(root)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.setup_kwargs = setup_kwargs or {}
        self.usage_client = usage_client

        self._server_socket: socket.socket | None = None
        self._client_socket: socket.socket | None = None
        self._protocol: IPIProtocol | None = None
        self._process: subprocess.Popen | None = None
        self._connected = False

        # Worker stdout/stderr are redirected to temp files (not pipes) so a
        # chatty model load can't fill the OS pipe buffer and block before the
        # worker connects. See _start_worker / _read_worker_output.
        self._stdout_file = None
        self._stderr_file = None

        # Holds the spawn_in_env context (staged wrapper + sidecar) open for
        # the life of the worker process; stop() closes it.
        self._spawn_stack: contextlib.ExitStack | None = None

        # Usage-record state: a session is a worker that actually connected.
        # stop() writes one anonymous record per session (see usage.py) and
        # resets these, so repeated stop() calls can't double-count.
        self._session_started_at: str | None = None
        self._session_started_monotonic: float | None = None
        self._n_calculations = 0

    def start(self):
        """Start the server and launch the worker process."""
        # Create server socket inside a fresh private (0700) directory
        self.socket_path = create_private_socket_path(self.socket_name)
        self._socket_dir = os.path.dirname(self.socket_path)
        self._server_socket = create_server_socket(self.socket_path, timeout=self.timeout)
        self._server_socket.listen(1)

        logger.debug("Server listening on %s", self.socket_path)

        # Launch worker process
        process = self._start_worker()

        logger.info(
            "Launched worker (PID %s) for %s/%s on %s",
            process.pid,
            self.env_name,
            self.checkpoint,
            self.device,
        )

        # Wait for worker to connect
        self._accept_connection()

    def _start_worker(self) -> subprocess.Popen:
        """Start worker using pre-built environment; return the worker process."""
        from .spawn import WORKER_WRAPPER, spawn_in_env

        self._spawn_stack = contextlib.ExitStack()
        spec = self._spawn_stack.enter_context(
            spawn_in_env(
                self.root,
                self.env_name,
                WORKER_WRAPPER,
                {
                    "checkpoint": self.checkpoint,
                    "checkpoint_path": self.checkpoint_path,
                    "device": self.device,
                    "socket_path": self.socket_path,
                    "setup_kwargs": self.setup_kwargs,
                },
                cache_root=self.cache_root,
            )
        )

        logger.debug("Spawning worker: %s", " ".join(spec.cmd))

        # Redirect worker output to temp files rather than pipes. The worker
        # loads the model in setup() *before* connecting to the socket; a noisy
        # load that exceeds the OS pipe buffer (~64 KB) would block on the write
        # and never connect, deadlocking _accept_connection. Regular files have
        # no such buffer limit, and we read them back for the post-mortem when
        # the worker fails. (The worker's own verbosity is controlled by the
        # ROOTSTOCK_WORKER_LOG env var — see worker_config.py.)
        self._stdout_file = tempfile.TemporaryFile()
        self._stderr_file = tempfile.TemporaryFile()

        self._process = subprocess.Popen(
            spec.cmd,
            env=spec.env,
            cwd=spec.cwd,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )
        return self._process

    def _accept_connection(self):
        """Accept connection from worker process.

        Bounded by ``self.timeout``: the socket timeouts below only start
        counting once a connection exists, so without a deadline here a
        worker that is alive but never connects — wedged in setup() on
        stalled filesystem I/O, say — would hang this loop forever with no
        way for the caller's timeout to fire (observed on Delta, workers
        blocked in Lustre reads of /work/hdd during model load).
        """
        server_socket = self._server_socket
        process = self._process
        assert server_socket is not None and process is not None  # set by start()

        # Use short timeout for accept so we can check if process died
        server_socket.settimeout(1.0)
        deadline = time.monotonic() + self.timeout
        next_heartbeat = 30.0

        while True:
            try:
                client_socket, addr = server_socket.accept()
                self._client_socket = client_socket
                break
            except TimeoutError:
                # Check if process died
                if process.poll() is not None:
                    raise self._worker_failure_error("Worker process died before connecting")
                waited = self.timeout - (deadline - time.monotonic())
                if waited >= next_heartbeat:
                    logger.info(
                        "Still waiting for worker %d to connect (%.0fs elapsed, "
                        "typically loading the model)",
                        process.pid,
                        waited,
                    )
                    next_heartbeat += 30.0
                if time.monotonic() >= deadline:
                    # Read the post-mortem (live output tails) before stop()
                    # closes the capture files, then tear the worker down —
                    # stop()'s bounded waits make that safe even when the
                    # worker is stuck in uninterruptible I/O.
                    error = self._worker_failure_error(
                        f"Worker did not connect within timeout ({self.timeout:g}s). "
                        "The worker never finished setup() — a slow or stalled "
                        "model load (cold cache, degraded filesystem). If the "
                        "load is genuinely slow, raise `timeout`"
                    )
                    self.stop()
                    raise error

        # Restore original timeout
        server_socket.settimeout(self.timeout)
        client_socket.settimeout(self.timeout)

        self._protocol = IPIProtocol(client_socket)
        self._connected = True

        from .manifest import now_iso

        self._session_started_at = now_iso()
        self._session_started_monotonic = time.monotonic()
        self._n_calculations = 0

        logger.info("Worker connected")

    def _read_worker_output(self) -> tuple[str, str]:
        """Read the worker's captured stdout/stderr from the temp files.

        Returns decoded ``(stdout, stderr)``. Empty strings when output was not
        captured (e.g. logging mode, where the worker inherits the parent fds).
        """

        def _drain(f) -> str:
            if f is None:
                return ""
            try:
                f.seek(0)
                return f.read().decode("utf-8", errors="replace")
            except (ValueError, OSError):
                return ""

        return _drain(self._stdout_file), _drain(self._stderr_file)

    def _worker_failure_error(self, context: str, exc: Exception | None = None) -> WorkerDiedError:
        """Build a post-mortem error for a worker failure.

        A worker that dies mid-``calculate`` (GPU OOM, batch-system kill)
        surfaces as a bare socket timeout or closed socket while the actual
        traceback sits unread in the captured output files — so read them on
        *any* worker failure and report the cause, not just the symptom.
        """
        if self._process is None:
            fate = "worker process was never started"
        else:
            # A dying worker delivers the socket error a beat before its exit
            # status is reapable (the fd closes during process teardown), so a
            # bare poll() here would misreport a dead worker as hung. Give it
            # a short grace period; a truly hung worker just waits it out.
            try:
                returncode = self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                returncode = None
            if returncode is not None:
                fate = f"worker process exited with code {returncode}"
            else:
                fate = "worker process is still running (hung, or blocked on the device?)"

        lines = [f"{context}: {fate}."]
        if exc is not None:
            lines.append(f"Cause: {type(exc).__name__}: {exc}")

        stdout, stderr = self._read_worker_output()
        captured = False
        for name, text in (("stdout", stdout), ("stderr", stderr)):
            text = text.strip()
            if text:
                captured = True
                lines.append(f"--- worker {name} (tail) ---\n{_tail(text)}")
        if not captured:
            lines.append("(worker produced no output)")
        return WorkerDiedError("\n".join(lines))

    def calculate(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        atomic_numbers: np.ndarray | None = None,
        pbc: list[bool] | None = None,
        info: dict | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate energy and forces for given atomic configuration.

        The worker returns to NEEDINIT after every force call, so the INIT
        payload (species, pbc, info) is re-sent each cycle — mid-run changes
        to any of them reach the worker.

        Args:
            positions: Nx3 array of atomic positions in Angstrom
            cell: 3x3 cell matrix in Angstrom
            atomic_numbers: Atomic numbers array (sent in INIT)
            pbc: Periodic boundary conditions [x, y, z] (sent in INIT)
            info: JSON-serializable ``atoms.info`` subset (sent in INIT) —
                model inputs like ``charge``/``spin``/``external_field`` that
                the worker applies to its Atoms.

        Returns:
            energy: Potential energy in eV
            forces: Nx3 forces in eV/Angstrom
            virial: 3x3 virial tensor in eV
        """
        if not self._connected or self._protocol is None:
            raise RuntimeError("Server not connected. Call start() first.")

        # A worker that dies mid-exchange (GPU OOM, batch-system kill) can't
        # report in-band; the failure shows up here as a socket timeout,
        # closed socket, or broken pipe. Turn that into a post-mortem that
        # includes the worker's captured output instead of a bare socket error.
        try:
            # Check worker status
            self._protocol.send_status()
            status = self._protocol.recv_status()

            if status == "NEEDINIT":
                # Send INIT with atomic species info
                numbers = atomic_numbers.tolist() if atomic_numbers is not None else None
                pbc_list = [bool(p) for p in pbc] if pbc is not None else [True, True, True]
                init_data = {"numbers": numbers, "pbc": pbc_list, "info": info or {}}
                init_bytes = json.dumps(init_data).encode("utf-8")
                self._protocol.send_init(bead_index=0, init_string=init_bytes)

                self._protocol.send_status()
                status = self._protocol.recv_status()

            if status != "READY":
                raise RuntimeError(f"Worker not ready, status: {status}")

            # Send positions
            self._protocol.send_posdata(cell, positions)

            # Check status - worker should now be calculating
            self._protocol.send_status()
            status = self._protocol.recv_status()

            if status != "HAVEDATA":
                raise RuntimeError(f"Worker failed to calculate, status: {status}")

            # Get results
            self._protocol.send_getforce()
            energy, forces, virial, extra = self._protocol.recv_forceready()
        except (TimeoutError, SocketClosed, OSError) as exc:
            raise self._worker_failure_error("Worker failed mid-calculation", exc) from exc

        error = _worker_error_from_extra(extra)
        if error is not None:
            raise RuntimeError(f"Worker calculation failed:\n{error}")

        self._n_calculations += 1
        return energy, forces, virial

    def _record_usage(self):
        """Write the session's anonymous usage record, if there was a session.

        Called from stop() while the session state is still intact. All the
        can't-fail guarantees live in record_session; the only job here is
        gathering the fields and making the write once per session.
        """
        if (
            self.usage_client is None
            or self._session_started_at is None
            or self._session_started_monotonic is None
        ):
            return
        started_at = self._session_started_at
        duration_s = time.monotonic() - self._session_started_monotonic
        self._session_started_at = None
        self._session_started_monotonic = None

        from .layout import resolve_cache_root

        record_session(
            root=self.root,
            cache_root=resolve_cache_root(self.root, explicit=self.cache_root),
            env_name=self.env_name,
            checkpoint=self.checkpoint,
            device=self.device,
            client=self.usage_client,
            started_at=started_at,
            duration_s=duration_s,
            n_calculations=self._n_calculations,
        )

    def stop(self):
        """Stop the server and terminate the worker process."""
        self._record_usage()

        if self._protocol is not None:
            try:
                self._protocol.send_exit()
            except (BrokenPipeError, SocketClosed):
                pass

        if self._client_socket is not None:
            self._client_socket.close()
            self._client_socket = None

        if self._server_socket is not None:
            self._server_socket.close()
            self._server_socket = None

        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    # A worker in uninterruptible sleep (D state — dead Lustre
                    # mount, swap thrash) ignores SIGKILL until its syscall
                    # returns. Waiting here would hang teardown forever, so
                    # abandon the process; the kernel reaps it when it wakes.
                    logger.warning(
                        "Worker process %d did not exit after SIGKILL "
                        "(likely stuck in uninterruptible I/O); "
                        "abandoning it and continuing teardown",
                        self._process.pid,
                    )
            self._process = None

        # Close worker output temp files
        for attr in ("_stdout_file", "_stderr_file"):
            f = getattr(self, attr)
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass
                setattr(self, attr, None)

        # Clean up the socket and its private directory
        if self._socket_dir is not None:
            shutil.rmtree(self._socket_dir, ignore_errors=True)
            self._socket_dir = None
            self.socket_path = None

        # Remove the staged wrapper + sidecar (the worker is gone by now)
        if self._spawn_stack is not None:
            self._spawn_stack.close()
            self._spawn_stack = None

        self._connected = False
        self._protocol = None

        logger.info("Server stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
