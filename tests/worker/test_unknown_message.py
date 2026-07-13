"""An unknown protocol message must kill the worker, not be skipped.

The wire format is untagged 12-byte headers followed by raw payloads, so an
unrecognized message means the stream is desynced; logging and continuing
(the old behavior) yields garbage reads or a hang.
"""

from __future__ import annotations

import socket

import pytest

from rootstock.protocol import IPIProtocol
from rootstock.worker import MLIPWorker


def _wired_worker() -> tuple[MLIPWorker, IPIProtocol]:
    """Worker pre-wired to one end of a socketpair; no real connect."""
    server_sock, worker_sock = socket.socketpair()
    worker = MLIPWorker(calculator=None, socket_path="/tmp/unused")
    worker._socket = worker_sock
    worker._protocol = IPIProtocol(worker_sock)
    worker._connect = lambda: None
    return worker, IPIProtocol(server_sock)


def test_unknown_message_raises():
    worker, server = _wired_worker()
    server.sendmsg("BOGUSMSG")
    with pytest.raises(RuntimeError, match="BOGUSMSG"):
        worker.run()


def test_known_messages_still_handled():
    """Sanity check: STATUS/EXIT round-trip still works after the change."""
    worker, server = _wired_worker()
    server.sendmsg("STATUS")
    server.sendmsg("EXIT")
    worker.run()  # returns cleanly on EXIT
    assert server.recvmsg() == "NEEDINIT"
