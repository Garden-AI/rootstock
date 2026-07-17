"""Worker output is read on *any* worker failure, not only pre-connect death.

A worker that dies mid-``calculate`` (GPU OOM, batch-system kill) can't use
the in-band FORCEREADY error channel; historically the server surfaced it as
a bare socket timeout/closed-socket while the traceback sat unread in the
captured output tempfiles. ``RootstockServer`` now attaches a post-mortem —
exit code plus stdout/stderr tails — to every worker failure.

Uses the faked-env harness from the pipe-deadlock test: ``bin/python``
symlinked to this interpreter, ``PYTHONPATH`` handed down so the worker can
import rootstock + ase.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from rootstock.server import RootstockServer, _tail

_CHECKPOINT = "postmortem-dummy"
_MID_CALC_MARKER = "MID-CALCULATE-BOOM-MARKER"
_SETUP_MARKER = "SETUP-BOOM-MARKER"
_EXIT_CODE = 19


def _fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_source: str) -> Path:
    import ase

    import rootstock

    pythonpath = os.pathsep.join(
        sorted({str(Path(m.__file__).resolve().parents[1]) for m in (ase, rootstock)})
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    env_dir = tmp_path / "envs" / "doomed"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(env_source)
    return tmp_path


@pytest.fixture
def mid_calculate_death_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Env whose calculator kills the worker process on the first force call —
    the shape of a GPU OOM: no exception to catch, no in-band error, just a
    dead peer."""
    return _fake_root(
        tmp_path,
        monkeypatch,
        f"CHECKPOINTS = {{{_CHECKPOINT!r}: 'dummy'}}\n\n"
        "def setup(checkpoint, device='cpu', **kwargs):\n"
        "    from ase.calculators.lj import LennardJones\n"
        "    class Doomed(LennardJones):\n"
        "        def calculate(self, *a, **k):\n"
        "            import os, sys\n"
        f"            sys.stderr.write({_MID_CALC_MARKER!r})\n"
        "            sys.stderr.flush()\n"
        f"            os._exit({_EXIT_CODE})\n"
        "    return Doomed()\n",
    )


@pytest.fixture
def setup_death_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Env whose setup() raises before the worker ever connects."""
    return _fake_root(
        tmp_path,
        monkeypatch,
        f"CHECKPOINTS = {{{_CHECKPOINT!r}: 'dummy'}}\n\n"
        "def setup(checkpoint, device='cpu', **kwargs):\n"
        f"    raise RuntimeError({_SETUP_MARKER!r})\n",
    )


def _calculate_h2(server: RootstockServer):
    positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    cell = np.zeros((3, 3))
    numbers = np.array([1, 1])
    return server.calculate(positions, cell, numbers, pbc=[False, False, False])


def test_mid_calculate_death_reports_exit_code_and_stderr(mid_calculate_death_root: Path):
    server = RootstockServer(
        env_name="doomed",
        checkpoint=_CHECKPOINT,
        device="cpu",
        root=mid_calculate_death_root,
        timeout=15.0,
        socket_name=f"postmortem_{os.getpid()}",
    )
    server.start()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _calculate_h2(server)
    finally:
        server.stop()

    message = str(excinfo.value)
    assert "Worker failed mid-calculation" in message
    assert f"exited with code {_EXIT_CODE}" in message
    assert _MID_CALC_MARKER in message  # the traceback-bearing stderr tail


def test_death_before_connect_reports_stderr_tail(setup_death_root: Path):
    server = RootstockServer(
        env_name="doomed",
        checkpoint=_CHECKPOINT,
        device="cpu",
        root=setup_death_root,
        timeout=15.0,
        socket_name=f"postmortem_pre_{os.getpid()}",
    )
    with pytest.raises(RuntimeError) as excinfo:
        server.start()
    server.stop()

    message = str(excinfo.value)
    assert "died before connecting" in message
    assert _SETUP_MARKER in message


def test_tail_truncates_long_output():
    text = "x" * 100_000 + "THE-END"
    tailed = _tail(text)
    assert len(tailed) < 10_000
    assert tailed.endswith("THE-END")
    assert "truncated" in tailed


def test_tail_passes_short_output_through():
    assert _tail("short") == "short"
