"""stop() must not hang on a worker process that never dies.

Observed on NCSA Delta (2026-07-23): a worker wedged in uninterruptible
sleep (D state — dead Lustre I/O, swap thrash) ignores SIGKILL until its
syscall returns, so the old ``self._process.wait()`` (no timeout) after
``kill()`` blocked teardown forever. User-visible symptom: the 600 s socket
timeout fired, but the WorkerDiedError post-mortem never printed because
the calculator's teardown hung first — "the timeout arg didn't actually
time out".

SIGKILL immunity can't be faked from userspace, so the harness combines a
real child process that ignores SIGTERM (proving terminate() alone can't
reap it) with a Popen wrapper whose ``wait(timeout=...)`` always raises
``TimeoutExpired`` — the observable behavior of a D-state process that
outlives both signals. A ``wait()`` call *without* a timeout fails the
test immediately: that is exactly the call that hangs on a real cluster.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from rootstock.server import RootstockServer

_SIGTERM_IMMUNE_WORKER = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "time.sleep(3600)\n"
)


class _UnkillablePopen:
    """Wraps a real SIGTERM-immune process, but reports every bounded wait
    as timing out and refuses unbounded waits — the shape of a D-state
    worker that survives SIGKILL."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self.pid = proc.pid
        self.kill_called = False

    def terminate(self):
        self._proc.terminate()  # real signal, really ignored

    def kill(self):
        # Deliberately not delivered to the real child: the simulated
        # worker survives SIGKILL. The fixture reaps it afterwards.
        self.kill_called = True

    def wait(self, timeout=None):
        if timeout is None:
            pytest.fail(
                "Popen.wait() called without a timeout during stop() — "
                "this hangs forever on a worker stuck in uninterruptible sleep"
            )
        raise subprocess.TimeoutExpired(cmd="fake-worker", timeout=timeout)


@pytest.fixture
def sigterm_immune_proc():
    proc = subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_IMMUNE_WORKER],
        stdout=subprocess.PIPE,
        text=True,
    )
    # Don't race the signal handler installation
    assert proc.stdout.readline().strip() == "ready"
    try:
        yield proc
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        proc.stdout.close()


def _bare_server(tmp_path: Path) -> RootstockServer:
    """A server that was never start()ed — stop() tolerates the unopened
    sockets/files, so we can graft a fake process straight onto it."""
    return RootstockServer(
        env_name="unused",
        checkpoint="unused",
        device="cpu",
        root=tmp_path,
        socket_name="unkillable-test",
    )


def test_stop_abandons_worker_that_survives_sigkill(tmp_path: Path, sigterm_immune_proc, caplog):
    server = _bare_server(tmp_path)
    fake = _UnkillablePopen(sigterm_immune_proc)
    server._process = fake

    with caplog.at_level(logging.WARNING, logger="rootstock.server"):
        server.stop()  # must return instead of blocking on wait()

    assert fake.kill_called  # escalated terminate -> kill before giving up
    assert server._process is None  # teardown completed past the process
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(str(fake.pid) in msg and "SIGKILL" in msg for msg in warnings)


def test_stop_reaps_worker_that_only_ignores_sigterm(tmp_path: Path, sigterm_immune_proc):
    # The ordinary escalation path, end to end with a real process: SIGTERM
    # is ignored, the 5 s bounded wait expires, SIGKILL actually lands.
    server = _bare_server(tmp_path)
    server._process = sigterm_immune_proc

    server.stop()

    assert server._process is None
    assert sigterm_immune_proc.returncode == -signal.SIGKILL
