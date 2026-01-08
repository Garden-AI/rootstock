"""
Minimal i-PI protocol implementation for Rootstock.

This is adapted from ASE's ase.calculators.socketio module.
The i-PI protocol is a simple binary protocol for communicating atomic
simulation data over sockets.

Protocol Overview:
- Commands are 12-byte ASCII strings (e.g., "STATUS", "POSDATA", "GETFORCE")
- Data is transmitted as raw numpy arrays (no JSON/msgpack serialization)
- Units are atomic units (Bohr, Hartree) - we convert to ASE units (Å, eV)

Reference: https://docs.ipi-code.org/
"""

import socket
import numpy as np

# ASE units conversion
BOHR_TO_ANGSTROM = 0.52917721067
HARTREE_TO_EV = 27.211386245988
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


class SocketClosed(Exception):
    """Raised when socket connection is closed."""
    pass


class IPIProtocol:
    """
    Communication using the i-PI protocol.
    
    This handles the low-level socket communication, including:
    - Sending/receiving fixed-length command strings
    - Sending/receiving numpy arrays as raw bytes
    - Unit conversions between i-PI (atomic units) and ASE (Å, eV)
    """
    
    def __init__(self, sock: socket.socket, log=None):
        """
        Initialize protocol handler.
        
        Args:
            sock: Connected socket object
            log: Optional file object for logging (useful for debugging)
        """
        self.socket = sock
        self.log = log
    
    def _log(self, *args):
        """Write to log if logging is enabled."""
        if self.log is not None:
            print(*args, file=self.log, flush=True)
    
    # -------------------------------------------------------------------------
    # Low-level send/receive
    # -------------------------------------------------------------------------
    
    def sendmsg(self, msg: str):
        """Send a 12-byte command string."""
        self._log(f"  sendmsg: {msg!r}")
        encoded = msg.encode('ascii').ljust(12)
        self.socket.sendall(encoded)
    
    def recvmsg(self) -> str:
        """Receive a 12-byte command string."""
        data = self._recvall(12)
        msg = data.rstrip().decode('ascii')
        self._log(f"  recvmsg: {msg!r}")
        return msg
    
    def _recvall(self, nbytes: int) -> bytes:
        """Receive exactly nbytes, handling partial reads."""
        chunks = []
        remaining = nbytes
        while remaining > 0:
            chunk = self.socket.recv(remaining)
            if len(chunk) == 0:
                raise SocketClosed("Socket closed while receiving data")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)
    
    def send_array(self, arr, dtype):
        """Send a numpy array as raw bytes."""
        buf = np.asarray(arr, dtype=dtype).tobytes()
        self._log(f"  send: {len(buf)} bytes ({dtype})")
        self.socket.sendall(buf)
    
    def recv_array(self, shape, dtype) -> np.ndarray:
        """Receive a numpy array from raw bytes."""
        arr = np.empty(shape, dtype=dtype)
        nbytes = arr.nbytes
        buf = self._recvall(nbytes)
        arr.flat[:] = np.frombuffer(buf, dtype=dtype)
        self._log(f"  recv: {nbytes} bytes ({dtype})")
        return arr
    
    # -------------------------------------------------------------------------
    # High-level protocol messages
    # -------------------------------------------------------------------------
    
    def send_status(self):
        """Send STATUS request."""
        self._log(" send_status")
        self.sendmsg("STATUS")
    
    def recv_status(self) -> str:
        """Receive status response (READY, HAVEDATA, or NEEDINIT)."""
        return self.recvmsg()
    
    def send_init(self, bead_index: int = 0, init_string: bytes = b'\x00'):
        """Send INIT message (required by some codes, often ignored)."""
        self._log(" send_init")
        self.sendmsg("INIT")
        self.send_array([bead_index], np.int32)
        self.send_array([len(init_string)], np.int32)
        self.send_array(np.frombuffer(init_string, dtype=np.byte), np.byte)
    
    def recv_init(self) -> tuple[int, bytes]:
        """Receive INIT message."""
        bead_index = self.recv_array(1, np.int32)[0]
        nbytes = self.recv_array(1, np.int32)[0]
        init_bytes = self.recv_array(nbytes, np.byte).tobytes()
        return bead_index, init_bytes
    
    def send_posdata(self, cell: np.ndarray, positions: np.ndarray):
        """
        Send atomic positions and cell.
        
        Args:
            cell: 3x3 cell matrix in Angstrom
            positions: Nx3 positions in Angstrom
        """
        self._log(" send_posdata")
        self.sendmsg("POSDATA")
        
        # Convert to atomic units and transpose (i-PI convention)
        cell_bohr = cell.T * ANGSTROM_TO_BOHR
        icell_bohr = np.linalg.pinv(cell).T / ANGSTROM_TO_BOHR
        positions_bohr = positions * ANGSTROM_TO_BOHR
        
        self.send_array(cell_bohr, np.float64)
        self.send_array(icell_bohr, np.float64)
        self.send_array([len(positions)], np.int32)
        self.send_array(positions_bohr, np.float64)
    
    def recv_posdata(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Receive atomic positions and cell.
        
        Returns:
            cell: 3x3 cell matrix in Angstrom
            positions: Nx3 positions in Angstrom
        """
        cell_bohr = self.recv_array((3, 3), np.float64).T.copy()
        icell_bohr = self.recv_array((3, 3), np.float64).T.copy()  # Not used
        natoms = self.recv_array(1, np.int32)[0]
        positions_bohr = self.recv_array((natoms, 3), np.float64)
        
        # Convert to ASE units
        cell = cell_bohr * BOHR_TO_ANGSTROM
        positions = positions_bohr * BOHR_TO_ANGSTROM
        return cell, positions
    
    def send_getforce(self):
        """Send GETFORCE request."""
        self._log(" send_getforce")
        self.sendmsg("GETFORCE")
    
    def recv_forceready(self) -> tuple[float, np.ndarray, np.ndarray, bytes]:
        """
        Receive force data after GETFORCE.
        
        Returns:
            energy: Potential energy in eV
            forces: Nx3 forces in eV/Angstrom  
            virial: 3x3 virial tensor in eV
            extra: Extra bytes (often empty)
        """
        msg = self.recvmsg()
        assert msg == "FORCEREADY", f"Expected FORCEREADY, got {msg}"
        
        energy_hartree = self.recv_array(1, np.float64)[0]
        natoms = self.recv_array(1, np.int32)[0]
        forces_au = self.recv_array((natoms, 3), np.float64)
        virial_au = self.recv_array((3, 3), np.float64).T.copy()
        nextra = self.recv_array(1, np.int32)[0]
        extra = self.recv_array(nextra, np.byte).tobytes() if nextra > 0 else b''
        
        # Convert to ASE units
        energy = energy_hartree * HARTREE_TO_EV
        forces = forces_au * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        virial = virial_au * HARTREE_TO_EV
        
        return energy, forces, virial, extra
    
    def send_forceready(self, energy: float, forces: np.ndarray, virial: np.ndarray,
                        extra: bytes = b'\x00'):
        """
        Send force data in response to GETFORCE.
        
        Args:
            energy: Potential energy in eV
            forces: Nx3 forces in eV/Angstrom
            virial: 3x3 virial tensor in eV
            extra: Extra bytes to send (minimum 1 byte)
        """
        self._log(" send_forceready")
        self.sendmsg("FORCEREADY")
        
        # Convert to atomic units
        energy_hartree = energy * EV_TO_HARTREE
        forces_au = forces / (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        virial_au = virial * EV_TO_HARTREE
        
        self.send_array([energy_hartree], np.float64)
        self.send_array([len(forces)], np.int32)
        self.send_array(forces_au, np.float64)
        self.send_array(virial_au.T, np.float64)
        
        # Always send at least 1 byte to avoid confusion with closed socket
        if len(extra) == 0:
            extra = b'\x00'
        self.send_array([len(extra)], np.int32)
        self.send_array(np.frombuffer(extra, dtype=np.byte), np.byte)
    
    def send_exit(self):
        """Send EXIT message to terminate connection."""
        self._log(" send_exit")
        self.sendmsg("EXIT")


# -----------------------------------------------------------------------------
# Socket creation helpers
# -----------------------------------------------------------------------------

def create_unix_socket_path(name: str) -> str:
    """
    Create path for Unix domain socket following i-PI convention.
    
    i-PI uses /tmp/ipi_<name> as the socket path.
    """
    return f"/tmp/ipi_{name}"


def create_server_socket(socket_path: str, timeout: float = None) -> socket.socket:
    """
    Create and bind a Unix domain socket server.
    
    Args:
        socket_path: Path for the socket file
        timeout: Optional timeout in seconds
        
    Returns:
        Bound socket ready for listen()
    """
    import os
    
    # Remove stale socket file if it exists
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    if timeout is not None:
        sock.settimeout(timeout)
    
    return sock


def connect_unix_socket(socket_path: str, timeout: float = None, 
                        max_retries: int = 50, retry_delay: float = 0.1) -> socket.socket:
    """
    Connect to a Unix domain socket server with retries.
    
    Args:
        socket_path: Path to the socket file
        timeout: Optional timeout for the connection
        max_retries: Maximum connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Connected socket
    """
    import time
    
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if timeout is not None:
        sock.settimeout(timeout)
    
    for attempt in range(max_retries):
        try:
            sock.connect(socket_path)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    
    raise RuntimeError(f"Failed to connect to {socket_path} after {max_retries} attempts")
