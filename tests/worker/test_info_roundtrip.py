"""atoms.info crosses the socket in the per-cycle INIT payload.

Drives a real RootstockServer.calculate() against a real MLIPWorker.run()
over a socketpair — the full protocol path, minus process spawning — and
asserts the worker's Atoms carries the forwarded info on every cycle,
including across an Atoms rebuild, and that workers tolerate INIT payloads
from servers that predate the field.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from rootstock.protocol import IPIProtocol
from rootstock.server import RootstockServer
from rootstock.worker import MLIPWorker

CELL = np.eye(3) * 10.0


class InfoRecordingCalculator:
    """Snapshots atoms.info at each energy evaluation."""

    def __init__(self):
        self.seen: list[dict] = []

    def get_potential_energy(self, atoms=None, **kwargs):
        self.seen.append(
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in atoms.info.items()}
        )
        return 1.0

    def get_forces(self, atoms=None):
        return np.zeros((len(atoms), 3))

    def reset(self):
        pass


@pytest.fixture
def linked_pair(tmp_path: Path):
    """A connected (server, worker-thread, calculator) triple."""
    server_sock, worker_sock = socket.socketpair()
    server_sock.settimeout(10.0)
    worker_sock.settimeout(10.0)

    calc = InfoRecordingCalculator()
    worker = MLIPWorker(calculator=calc, socket_path="/tmp/unused")
    worker._socket = worker_sock
    worker._protocol = IPIProtocol(worker_sock)
    worker._connect = lambda: worker._protocol

    server = RootstockServer(
        env_name="fake",
        checkpoint="fake-1",
        device="cpu",
        root=tmp_path,
        usage_client=None,
    )
    server._protocol = IPIProtocol(server_sock)
    server._connected = True

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        yield server, calc
    finally:
        try:
            server._protocol.sendmsg("EXIT")
        except OSError:
            pass
        thread.join(timeout=10.0)
        server_sock.close()


def test_info_reaches_worker_and_updates_per_cycle(linked_pair):
    server, calc = linked_pair
    positions = np.zeros((2, 3))
    numbers = np.array([1, 1])

    server.calculate(positions, CELL, numbers, [True] * 3, info={"charge": 0, "spin": 1})
    server.calculate(
        positions,
        CELL,
        numbers,
        [True] * 3,
        info={"charge": 1, "spin": 2, "external_field": [0.0, 0.0, 0.5]},
    )

    assert calc.seen[0] == {"charge": 0, "spin": 1}
    assert calc.seen[1]["charge"] == 1
    assert calc.seen[1]["spin"] == 2
    field = calc.seen[1]["external_field"]
    assert isinstance(field, np.ndarray)
    np.testing.assert_allclose(field, [0.0, 0.0, 0.5])


def test_info_survives_atoms_rebuild(linked_pair):
    server, calc = linked_pair
    positions = np.zeros((2, 3))

    server.calculate(positions, CELL, np.array([1, 1]), [True] * 3, info={"charge": 1})
    # Same-count composition change forces the worker to rebuild its Atoms
    server.calculate(positions, CELL, np.array([1, 8]), [True] * 3, info={"charge": 1})

    assert calc.seen[0] == {"charge": 1}
    assert calc.seen[1] == {"charge": 1}


def test_omitted_info_is_empty(linked_pair):
    server, calc = linked_pair
    positions = np.zeros((2, 3))

    server.calculate(positions, CELL, np.array([1, 1]), [True] * 3)
    assert calc.seen[0] == {}


def test_worker_tolerates_init_without_info_field():
    """An INIT from a server that predates the info field must still work."""
    server_sock, worker_sock = socket.socketpair()
    calc = InfoRecordingCalculator()
    worker = MLIPWorker(calculator=calc, socket_path="/tmp/unused")
    worker._socket = worker_sock
    worker._protocol = IPIProtocol(worker_sock)
    worker._connect = lambda: worker._protocol

    server = IPIProtocol(server_sock)
    old_init = json.dumps({"numbers": [1, 1], "pbc": [True, True, True]}).encode("utf-8")
    server.send_init(bead_index=0, init_string=old_init)
    server.send_posdata(CELL, np.zeros((2, 3)))
    server.send_getforce()
    server.sendmsg("EXIT")

    worker.run()

    energy, _, _, _ = server.recv_forceready()
    assert energy == pytest.approx(1.0)
    assert calc.seen[0] == {}
