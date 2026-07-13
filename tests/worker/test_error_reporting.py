"""Calculation failures are reported in-band via FORCEREADY extras.

A failing calculation (bad structure, GPU OOM, ...) used to kill the worker
with the traceback stranded in a stderr tempfile the server never read. The
worker now stays alive and ships the traceback as JSON in the otherwise
unused FORCEREADY extra field.
"""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest

from rootstock.protocol import IPIProtocol
from rootstock.worker import MLIPWorker

CELL = np.eye(3) * 10.0
POSITIONS = np.zeros((2, 3))
INIT_JSON = json.dumps({"numbers": [1, 1], "pbc": [True, True, True]}).encode("utf-8")


class ExplodingCalculator:
    def get_potential_energy(self, atoms=None, **kwargs):
        raise ValueError("boom: CUDA out of memory")


class StubCalculator:
    def get_potential_energy(self, atoms=None, **kwargs):
        return 1.25

    def get_forces(self, atoms=None):
        return np.ones((len(atoms), 3))


def _run_one_force_call(calculator) -> tuple[float, np.ndarray, np.ndarray, bytes]:
    """Drive a worker through INIT -> POSDATA -> GETFORCE -> EXIT."""
    server_sock, worker_sock = socket.socketpair()
    worker = MLIPWorker(socket_name="test", calculator=calculator)
    worker._socket = worker_sock
    worker._protocol = IPIProtocol(worker_sock)
    worker._connect = lambda: None

    server = IPIProtocol(server_sock)
    server.send_init(bead_index=0, init_string=INIT_JSON)
    server.send_posdata(CELL, POSITIONS)
    server.send_getforce()
    server.sendmsg("EXIT")

    worker.run()  # processes the queued messages, then exits cleanly

    return server.recv_forceready()


def test_calculation_error_lands_in_extras():
    energy, forces, virial, extra = _run_one_force_call(ExplodingCalculator())

    payload = json.loads(extra.decode("utf-8"))
    assert "boom: CUDA out of memory" in payload["error"]
    assert "Traceback" in payload["error"]
    # Placeholder results so the wire framing stays intact
    assert energy == 0.0
    np.testing.assert_array_equal(forces, np.zeros((2, 3)))
    np.testing.assert_array_equal(virial, np.zeros((3, 3)))


def test_successful_calculation_sends_no_error():
    energy, forces, virial, extra = _run_one_force_call(StubCalculator())

    assert energy == pytest.approx(1.25)
    np.testing.assert_allclose(forces, np.ones((2, 3)))
    assert extra in (b"", b"\x00")  # protocol pads empty extras to one byte


def test_worker_survives_a_failed_calculation():
    """The worker must keep serving after reporting an error."""
    server_sock, worker_sock = socket.socketpair()

    calc = StubCalculator()
    fail_once = {"armed": True}

    def flaky_energy(atoms=None, **kwargs):
        if fail_once.pop("armed", False):
            raise ValueError("transient failure")
        return 1.25

    calc.get_potential_energy = flaky_energy

    worker = MLIPWorker(socket_name="test", calculator=calc)
    worker._socket = worker_sock
    worker._protocol = IPIProtocol(worker_sock)
    worker._connect = lambda: None

    server = IPIProtocol(server_sock)
    server.send_init(bead_index=0, init_string=INIT_JSON)
    server.send_posdata(CELL, POSITIONS)
    server.send_getforce()
    # second cycle after the failure
    server.send_init(bead_index=0, init_string=INIT_JSON)
    server.send_posdata(CELL, POSITIONS)
    server.send_getforce()
    server.sendmsg("EXIT")

    worker.run()

    _, _, _, extra_first = server.recv_forceready()
    energy, _, _, extra_second = server.recv_forceready()
    assert b"transient failure" in extra_first
    assert extra_second in (b"", b"\x00")
    assert energy == pytest.approx(1.25)
