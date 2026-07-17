"""
Socket server for Rootstock.

This runs in the main process and acts as an i-PI server,
sending atomic positions and receiving forces from a worker process.
"""

import json
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .protocol import (
    IPIProtocol,
    SocketClosed,
    create_private_socket_path,
    create_server_socket,
)


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
        log=None,
        timeout: float = 60.0,
        setup_kwargs: dict | None = None,
    ):
        """
        Initialize the server.

        Args:
            env_name: Name of pre-built environment (e.g., "mace")
            checkpoint: Canonical checkpoint id passed to the env's setup()
            device: Device string to pass to setup()
            socket_name: Name for the Unix socket. The socket is created as
                ipi_<name> inside a fresh private (0700) temp directory on
                start(), so it is unreachable by other local users.
            root: Root directory for environments and cache (required)
            log: Optional file object for protocol logging
            timeout: Socket timeout in seconds
            setup_kwargs: Extra keyword arguments forwarded to setup()
        """
        if root is None:
            raise ValueError("root is required for pre-built environments")

        self.socket_name = socket_name
        # Created by start(): the socket lives in a private per-server temp
        # directory that stop() removes.
        self.socket_path: str | None = None
        self._socket_dir: str | None = None
        self.log = log
        self.timeout = timeout

        self.env_name = env_name
        self.checkpoint = checkpoint
        self.device = device
        self.root = Path(root)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.setup_kwargs = setup_kwargs or {}

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

        # Track INIT state
        self._init_sent = False
        self._init_numbers: list[int] | None = None
        self._init_pbc: list[bool] | None = None

        # Environment manager
        self._env_manager = None
        self._wrapper_path: Path | None = None

    def start(self):
        """Start the server and launch the worker process."""
        # Create server socket inside a fresh private (0700) directory
        self.socket_path = create_private_socket_path(self.socket_name)
        self._socket_dir = os.path.dirname(self.socket_path)
        self._server_socket = create_server_socket(self.socket_path, timeout=self.timeout)
        self._server_socket.listen(1)

        if self.log:
            print(f"Server listening on {self.socket_path}", file=self.log, flush=True)

        # Launch worker process
        self._start_worker()

        if self.log:
            print(f"Launched worker process (PID {self._process.pid})", file=self.log, flush=True)

        # Wait for worker to connect
        self._accept_connection()

    def _start_worker(self):
        """Start worker using pre-built environment."""
        from .environment import EnvironmentManager

        # Create environment manager
        self._env_manager = EnvironmentManager(root=self.root, cache_root=self.cache_root)

        # Generate wrapper script
        self._wrapper_path = self._env_manager.generate_wrapper(
            env_name=self.env_name,
            checkpoint=self.checkpoint,
            device=self.device,
            socket_path=self.socket_path,
            setup_kwargs=self.setup_kwargs,
        )

        # Get spawn command and environment
        cmd = self._env_manager.get_spawn_command(self.env_name, self._wrapper_path)
        env = self._env_manager.get_environment_variables()

        if self.log:
            print(f"Spawning worker: {' '.join(cmd)}", file=self.log, flush=True)

        # Redirect worker output to temp files rather than pipes. The worker
        # loads the model in setup() *before* connecting to the socket; a noisy
        # load that exceeds the OS pipe buffer (~64 KB) would block on the write
        # and never connect, deadlocking _accept_connection. Regular files have
        # no such buffer limit, and we can still read them back to report errors
        # if the worker dies. When logging, inherit the parent's fds as before.
        if not self.log:
            self._stdout_file = tempfile.TemporaryFile()
            self._stderr_file = tempfile.TemporaryFile()

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._stdout_file,
            stderr=self._stderr_file,
        )

    def _accept_connection(self):
        """Accept connection from worker process."""
        # Use short timeout for accept so we can check if process died
        self._server_socket.settimeout(1.0)

        while True:
            try:
                self._client_socket, addr = self._server_socket.accept()
                break
            except TimeoutError:
                # Check if process died
                if self._process.poll() is not None:
                    raise self._worker_failure_error("Worker process died before connecting")

        # Restore original timeout
        self._server_socket.settimeout(self.timeout)
        self._client_socket.settimeout(self.timeout)

        self._protocol = IPIProtocol(self._client_socket, log=self.log)
        self._connected = True

        if self.log:
            print("Worker connected", file=self.log, flush=True)

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

    def _worker_failure_error(self, context: str, exc: Exception | None = None) -> RuntimeError:
        """Build a post-mortem error for a worker failure.

        A worker that dies mid-``calculate`` (GPU OOM, batch-system kill)
        surfaces as a bare socket timeout or closed socket while the actual
        traceback sits unread in the captured output files — so read them on
        *any* worker failure and report the cause, not just the symptom.
        """
        if self._process is None:
            fate = "worker process was never started"
        elif self._process.poll() is not None:
            fate = f"worker process exited with code {self._process.returncode}"
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
            note = "(no worker output captured"
            if self.log:
                note += " — logging mode inherits the parent's stdio"
            lines.append(note + ")")
        return RuntimeError("\n".join(lines))

    def calculate(
        self,
        positions: np.ndarray,
        cell: np.ndarray,
        atomic_numbers: np.ndarray | None = None,
        pbc: list[bool] | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate energy and forces for given atomic configuration.

        Args:
            positions: Nx3 array of atomic positions in Angstrom
            cell: 3x3 cell matrix in Angstrom
            atomic_numbers: Atomic numbers array (sent in INIT on first call)
            pbc: Periodic boundary conditions [x, y, z] (sent in INIT on first call)

        Returns:
            energy: Potential energy in eV
            forces: Nx3 forces in eV/Angstrom
            virial: 3x3 virial tensor in eV
        """
        if not self._connected:
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
                init_data = {
                    "numbers": atomic_numbers.tolist() if atomic_numbers is not None else None,
                    "pbc": [bool(p) for p in pbc] if pbc is not None else [True, True, True],
                }
                init_bytes = json.dumps(init_data).encode("utf-8")
                self._protocol.send_init(bead_index=0, init_string=init_bytes)

                # Track what we sent
                self._init_sent = True
                self._init_numbers = init_data["numbers"]
                self._init_pbc = init_data["pbc"]

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

        return energy, forces, virial

    def stop(self):
        """Stop the server and terminate the worker process."""
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
                self._process.wait()
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

        # Clean up wrapper script
        if self._wrapper_path is not None:
            try:
                self._wrapper_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._wrapper_path = None

        # Clean up environment manager
        if self._env_manager is not None:
            self._env_manager.cleanup()
            self._env_manager = None

        self._connected = False
        self._protocol = None

        if self.log:
            print("Server stopped", file=self.log, flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
