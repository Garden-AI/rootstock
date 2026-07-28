"""start() must not hang on a worker that is alive but never connects.

Observed on NCSA Delta (2026-07-24, issue #160): a worker blocked in Lustre
sync I/O (`wchan cl_sync_io_wait`) during model load never reached the
socket connect, and the parent sat in ``_accept_connection``'s accept()
poll loop forever. The calculator's ``timeout`` only ever applied to the
*connected* socket, so from the user's chair "the timeout arg didn't
actually time out" — no matter what value was passed.

``_accept_connection`` now carries a deadline of ``self.timeout``: on
expiry it raises a WorkerDiedError post-mortem (with the worker's live
output tails) and tears the worker down.

Uses the faked-env harness from the post-mortem tests: ``bin/python``
symlinked to this interpreter, ``PYTHONPATH`` handed down so the worker
can import rootstock + ase.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from rootstock.server import RootstockServer, WorkerDiedError

_CHECKPOINT = "never-connects-dummy"
_SETUP_MARKER = "SETUP-STARTED-MARKER"


@pytest.fixture
def never_connecting_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Env whose setup() announces itself and then blocks forever — the
    shape of a model load stuck on stalled filesystem I/O. The worker only
    connects *after* setup() returns, so it never does."""
    import ase

    import rootstock

    pythonpath = os.pathsep.join(
        sorted({str(Path(m.__file__).resolve().parents[1]) for m in (ase, rootstock)})
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    env_dir = tmp_path / "envs" / "stalled"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(
        f"CHECKPOINTS = {{{_CHECKPOINT!r}: 'dummy'}}\n\n"
        "def setup(checkpoint, device='cpu', **kwargs):\n"
        "    import sys, time\n"
        f"    sys.stderr.write({_SETUP_MARKER!r})\n"
        "    sys.stderr.flush()\n"
        "    time.sleep(3600)\n"
    )
    return tmp_path


def test_start_times_out_when_worker_never_connects(never_connecting_root: Path):
    server = RootstockServer(
        env_name="stalled",
        checkpoint=_CHECKPOINT,
        device="cpu",
        root=never_connecting_root,
        timeout=3.0,
        socket_name=f"stalled_{os.getpid()}",
    )

    began = time.monotonic()
    with pytest.raises(WorkerDiedError) as excinfo:
        server.start()
    elapsed = time.monotonic() - began

    # Bounded: the old code looped in accept() forever. Allow slack for
    # worker spawn + the 1 s accept poll granularity + teardown.
    assert elapsed < 30.0

    message = str(excinfo.value)
    assert "did not connect within timeout" in message
    assert "still running" in message  # fate: alive-but-stuck, not dead
    assert _SETUP_MARKER in message  # live-read output tail made it in

    # The stuck worker was torn down, not orphaned.
    assert server._process is None
