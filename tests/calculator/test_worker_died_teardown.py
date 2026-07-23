"""A failing teardown must not mask the WorkerDiedError post-mortem.

Companion to tests/server/test_stop_unkillable_worker.py (NCSA Delta,
2026-07-23): calculate()'s WorkerDiedError handler used to call close()
before re-raising, so a stop() that hung or raised swallowed the one
exception carrying the worker's exit code and output tails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from rootstock.calculator import RootstockCalculator
from rootstock.server import WorkerDiedError

_ENV_SOURCE = """\
CHECKPOINTS = {"uma-s-1p1": "uma-s-1p1"}


def setup(checkpoint, device="cuda"):
    return None
"""

_POSTMORTEM = "worker exited with code 19\nMID-CALCULATE-BOOM"


class _DoomedServer:
    """calculate() reports a dead worker; stop() itself then blows up."""

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def calculate(self, *args, **kwargs):
        raise WorkerDiedError(_POSTMORTEM)

    def stop(self):
        raise RuntimeError("teardown failed too")


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    env_dir = tmp_path / "root" / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE)
    return tmp_path / "root"


def test_worker_died_error_survives_failing_close(fake_root, monkeypatch, caplog):
    monkeypatch.setattr("rootstock.calculator.RootstockServer", _DoomedServer)
    calc = RootstockCalculator(checkpoint="uma-s-1p1", root=fake_root, device="cpu")
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.calc = calc

    with pytest.raises(WorkerDiedError, match="MID-CALCULATE-BOOM"):
        calc.calculate(atoms)

    # The broken server was still discarded, so the next calculation
    # would build a fresh one instead of reusing the dead worker.
    assert calc._server is None
