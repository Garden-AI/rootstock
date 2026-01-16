"""
IPC overhead benchmark.

Measures the pure IPC overhead of the i-PI protocol, without actual MLIP calculation.
"""

import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from ase import Atoms

from .direct import BenchmarkResult


def benchmark_ipc_overhead_only(
    atoms: Atoms,
    n_calls: int = 1000,
    n_warmup: int = 10,
) -> BenchmarkResult:
    """
    Benchmark ONLY the IPC overhead, without actual MLIP calculation.

    Uses a mock worker that returns zeros instantly.
    This isolates the IPC overhead from MLIP compute time.

    Args:
        atoms: ASE Atoms object (only used for shape)
        n_calls: Number of round-trips to time
        n_warmup: Number of warmup calls

    Returns:
        BenchmarkResult with timing data
    """
    # We'll create a minimal mock worker inline
    repo_root = Path(__file__).parent.parent
    mock_worker_code = f'''
import sys
import numpy as np
sys.path.insert(0, "{repo_root}")
from rootstock.protocol import (
    IPIProtocol, connect_unix_socket, create_unix_socket_path
)

socket_name = sys.argv[1]
n_atoms = int(sys.argv[2])

sock = connect_unix_socket(create_unix_socket_path(socket_name))
protocol = IPIProtocol(sock)

while True:
    msg = protocol.recvmsg()

    if msg == "EXIT":
        break
    elif msg == "STATUS":
        protocol.sendmsg("READY")
    elif msg == "POSDATA":
        cell, positions = protocol.recv_posdata()
        # Mock calculation - instant return of zeros
        energy = 0.0
        forces = np.zeros_like(positions)
        virial = np.zeros((3, 3))
        # Wait for GETFORCE request
        msg2 = protocol.recvmsg()
        if msg2 == "STATUS":
            protocol.sendmsg("HAVEDATA")
            msg3 = protocol.recvmsg()
            if msg3 == "GETFORCE":
                protocol.send_forceready(energy, forces, virial)

sock.close()
'''

    # Write mock worker to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(mock_worker_code)
        mock_worker_path = f.name

    try:
        import os

        from rootstock.protocol import (
            IPIProtocol,
            create_server_socket,
            create_unix_socket_path,
        )

        socket_name = f"rootstock_mock_{uuid.uuid4().hex[:8]}"
        socket_path = create_unix_socket_path(socket_name)

        # Create server
        server_sock = create_server_socket(socket_path)
        server_sock.listen(1)

        # Launch mock worker
        proc = subprocess.Popen([
            sys.executable, mock_worker_path, socket_name, str(len(atoms))
        ])

        # Accept connection
        client_sock, _ = server_sock.accept()
        protocol = IPIProtocol(client_sock)

        # Warmup
        positions = atoms.positions.copy()
        cell = np.array(atoms.cell)

        for _ in range(n_warmup):
            protocol.send_status()
            protocol.recv_status()
            protocol.send_posdata(cell, positions)
            protocol.send_status()
            protocol.recv_status()
            protocol.send_getforce()
            protocol.recv_forceready()

        # Timed runs
        times_ms = []
        for _ in range(n_calls):
            t0 = time.perf_counter()

            protocol.send_status()
            protocol.recv_status()
            protocol.send_posdata(cell, positions)
            protocol.send_status()
            protocol.recv_status()
            protocol.send_getforce()
            protocol.recv_forceready()

            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

        # Cleanup
        protocol.send_exit()
        client_sock.close()
        server_sock.close()
        proc.wait()

        if os.path.exists(socket_path):
            os.unlink(socket_path)

    finally:
        import os
        os.unlink(mock_worker_path)

    return BenchmarkResult(
        name="ipc_overhead_only",
        n_atoms=len(atoms),
        n_calls=n_calls,
        times_ms=times_ms,
    )
