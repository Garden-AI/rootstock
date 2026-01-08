"""
Socket server for Rootstock.

This runs in the main process and acts as an i-PI server,
sending atomic positions and receiving forces from a worker process.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .protocol import (
    IPIProtocol,
    SocketClosed,
    create_server_socket,
    create_unix_socket_path,
)


class RootstockServer:
    """
    Server that communicates with an MLIP worker process via i-PI protocol.
    
    The server:
    1. Creates a Unix domain socket
    2. Launches a worker subprocess
    3. Accepts the worker's connection
    4. Sends positions, receives forces
    
    Example:
        with RootstockServer(mlip="mace-mp-0") as server:
            energy, forces, virial = server.calculate(positions, cell, numbers)
    """
    
    def __init__(
        self,
        socket_name: str = "rootstock",
        worker_script: Optional[str] = None,
        worker_python: Optional[str] = None,
        log=None,
        timeout: float = 60.0,
    ):
        """
        Initialize the server.
        
        Args:
            socket_name: Name for the Unix socket (will be /tmp/ipi_<name>)
            worker_script: Path to worker script. If None, uses default worker.py
            worker_python: Python executable for worker. If None, uses sys.executable
            log: Optional file object for protocol logging
            timeout: Socket timeout in seconds
        """
        self.socket_name = socket_name
        self.socket_path = create_unix_socket_path(socket_name)
        self.worker_script = worker_script or self._default_worker_script()
        self.worker_python = worker_python or sys.executable
        self.log = log
        self.timeout = timeout
        
        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._protocol: Optional[IPIProtocol] = None
        self._process: Optional[subprocess.Popen] = None
        self._connected = False
    
    def _default_worker_script(self) -> str:
        """Get path to the default worker script."""
        return str(Path(__file__).parent / "worker.py")
    
    def start(self):
        """Start the server and launch the worker process."""
        # Create server socket
        self._server_socket = create_server_socket(self.socket_path, timeout=self.timeout)
        self._server_socket.listen(1)
        
        if self.log:
            print(f"Server listening on {self.socket_path}", file=self.log, flush=True)
        
        # Launch worker process
        cmd = [self.worker_python, self.worker_script, "--socket", self.socket_name]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not self.log else None,
            stderr=subprocess.PIPE if not self.log else None,
        )
        
        if self.log:
            print(f"Launched worker process (PID {self._process.pid})", file=self.log, flush=True)
        
        # Wait for worker to connect
        self._accept_connection()
    
    def _accept_connection(self):
        """Accept connection from worker process."""
        # Use short timeout for accept so we can check if process died
        self._server_socket.settimeout(1.0)
        
        while True:
            try:
                self._client_socket, addr = self._server_socket.accept()
                break
            except socket.timeout:
                # Check if process died
                if self._process.poll() is not None:
                    stdout, stderr = self._process.communicate()
                    raise RuntimeError(
                        f"Worker process died with code {self._process.returncode}.\n"
                        f"stdout: {stdout}\nstderr: {stderr}"
                    )
        
        # Restore original timeout
        self._server_socket.settimeout(self.timeout)
        self._client_socket.settimeout(self.timeout)
        
        self._protocol = IPIProtocol(self._client_socket, log=self.log)
        self._connected = True
        
        if self.log:
            print("Worker connected", file=self.log, flush=True)
    
    def calculate(
        self, 
        positions: np.ndarray, 
        cell: np.ndarray,
        atomic_numbers: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate energy and forces for given atomic configuration.
        
        Args:
            positions: Nx3 array of atomic positions in Angstrom
            cell: 3x3 cell matrix in Angstrom
            atomic_numbers: Optional atomic numbers (sent on first call only)
            
        Returns:
            energy: Potential energy in eV
            forces: Nx3 forces in eV/Angstrom
            virial: 3x3 virial tensor in eV
        """
        if not self._connected:
            raise RuntimeError("Server not connected. Call start() first.")
        
        # Check worker status
        self._protocol.send_status()
        status = self._protocol.recv_status()
        
        if status == "NEEDINIT":
            # Some implementations require INIT before first calculation
            self._protocol.send_init()
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
        
        # Clean up socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
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
