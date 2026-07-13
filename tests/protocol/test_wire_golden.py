"""Golden wire-bytes test: the i-PI dialect byte stream is a frozen ABI.

Every built env pins its own rootstock, so the worker side of protocol.py
is frozen at whatever bytes 1.0 ships. These goldens pin the exact wire
bytes of every message for fixed inputs. If this test fails, the change
breaks compatibility with every deployed environment — fix the change, not
the test. See "Worker compatibility policy" in docs/development.md.

Goldens were captured from the implementation at the time this test was
written (little-endian float64/int32 payloads; the unit-conversion
constants in protocol.py are part of the frozen contract).
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from rootstock.protocol import IPIProtocol

# --- Fixed inputs -----------------------------------------------------------

INIT_STRING = b'{"numbers": [1, 8], "pbc": [true, true, true]}'
CELL = np.diag([10.0, 10.0, 10.0])  # Angstrom
POSITIONS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])  # Angstrom
ENERGY = 1.25  # eV
FORCES = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # eV/Angstrom
VIRIAL = np.diag([2.0, 2.0, 2.0])  # eV

# --- Golden bytes (frozen — do not regenerate to make a failure pass) -------

MSG_GOLDENS = {
    "STATUS": bytes.fromhex("535441545553202020202020"),
    "NEEDINIT": bytes.fromhex("4e454544494e495420202020"),
    "READY": bytes.fromhex("524541445920202020202020"),
    "HAVEDATA": bytes.fromhex("484156454441544120202020"),
    "GETFORCE": bytes.fromhex("474554464f52434520202020"),
    "EXIT": bytes.fromhex("455849542020202020202020"),
}

INIT_GOLDEN = bytes.fromhex(
    "494e49542020202020202020000000002e0000007b226e756d62657273223a205b312c"
    "20385d2c2022706263223a205b747275652c20747275652c20747275655d7d"
)

POSDATA_GOLDEN = bytes.fromhex(
    "504f53444154412020202020b461e0e9b2e53240000000000000000000000000000000"
    "000000000000000000b461e0e9b2e5324000000000000000000000000000000000"
    "0000000000000000b461e0e9b2e53240a225b9120818ab3f0000000000000000000000"
    "00000000000000000000000000a225b9120818ab3f0000000000000000000000000000"
    "00000000000000000000a225b9120818ab3f02000000000000000000000000000000000000"
    "00000000000000000086cf3376513cee3f86cf3376513cee3f86cf3376513cee3f"
)

FORCEREADY_GOLDEN = bytes.fromhex(
    "464f52434552454144592020358c3c4a0285a73f020000002d3c5e9fe3e9933f2d3c5e"
    "9fe3e9a33f435a0d6fd5dead3f2d3c5e9fe3e9b33f38cb35875ce4b83f435a0d6fd5de"
    "bd3fc409caa1ced0b23f000000000000000000000000000000000000000000000000c4"
    "09caa1ced0b23f000000000000000000000000000000000000000000000000c409caa1"
    "ced0b23f0100000000"
)

# --- Helpers -----------------------------------------------------------------


def _capture(send_fn) -> bytes:
    """Run send_fn against one end of a socketpair, return the raw bytes."""
    a, b = socket.socketpair()
    try:
        send_fn(IPIProtocol(a))
        a.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = b.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        a.close()
        b.close()


def _feed(data: bytes) -> IPIProtocol:
    """Return a protocol whose socket already holds data to be read."""
    a, b = socket.socketpair()
    a.sendall(data)
    a.shutdown(socket.SHUT_WR)
    return IPIProtocol(b)


# --- Send side: exact bytes on the wire --------------------------------------


@pytest.mark.parametrize("msg", sorted(MSG_GOLDENS))
def test_command_framing(msg):
    """Commands are exactly 12 bytes, ASCII, right-padded with spaces."""
    assert _capture(lambda p: p.sendmsg(msg)) == MSG_GOLDENS[msg]


def test_init_bytes():
    got = _capture(lambda p: p.send_init(bead_index=0, init_string=INIT_STRING))
    assert got == INIT_GOLDEN


def test_posdata_bytes():
    got = _capture(lambda p: p.send_posdata(CELL, POSITIONS))
    assert got == POSDATA_GOLDEN


def test_forceready_bytes():
    got = _capture(lambda p: p.send_forceready(ENERGY, FORCES, VIRIAL))
    assert got == FORCEREADY_GOLDEN


# --- Receive side: golden bytes parse back to the inputs ---------------------


def test_init_roundtrip():
    proto = _feed(INIT_GOLDEN)
    assert proto.recvmsg() == "INIT"
    bead_index, init_bytes = proto.recv_init()
    assert bead_index == 0
    assert init_bytes == INIT_STRING


def test_posdata_roundtrip():
    proto = _feed(POSDATA_GOLDEN)
    assert proto.recvmsg() == "POSDATA"
    cell, positions = proto.recv_posdata()
    np.testing.assert_allclose(cell, CELL)
    np.testing.assert_allclose(positions, POSITIONS)


def test_forceready_roundtrip():
    proto = _feed(FORCEREADY_GOLDEN)
    energy, forces, virial, extra = proto.recv_forceready()
    assert energy == pytest.approx(ENERGY)
    np.testing.assert_allclose(forces, FORCES)
    np.testing.assert_allclose(virial, VIRIAL)
    assert extra == b"\x00"  # empty extras are padded to one byte
