"""Calculator defaults and lifecycle: timeouts, root fallback, crash recovery.

Pre-1.0 decisions from #108:
- the calculator exposes ``timeout`` and both it and the server default to
  600 s — the envelope checkpoint verification already exercises — instead
  of a 60 s server default the first torch.compile would blow through;
- with neither ``cluster`` nor ``root``, the calculator falls back to
  ROOTSTOCK_ROOT and then the user config file, exactly like the CLI;
- a worker death mid-calculation (``WorkerDiedError``) tears the server
  down so the *next* calculation starts fresh — one GPU OOM no longer
  permanently bricks the calculator instance. No automatic retry.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest
from ase.build import molecule

from rootstock.calculator import RootstockCalculator
from rootstock.server import RootstockServer, WorkerDiedError

_CHECKPOINT = "lifecycle-dummy"

_ENV_SOURCE = f'''\
CHECKPOINTS = {{"{_CHECKPOINT}": "dummy"}}


def setup(checkpoint, device="cuda"):
    return None
'''


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    env_dir = tmp_path / "envs" / "lj"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE)
    return tmp_path


# --- timeouts ---------------------------------------------------------------


def test_server_default_timeout_matches_verification():
    assert inspect.signature(RootstockServer).parameters["timeout"].default == 600.0


def test_calculator_default_timeout_matches_verification(fake_root: Path):
    calc = RootstockCalculator(checkpoint=_CHECKPOINT, root=fake_root, device="cpu")
    assert calc.timeout == 600.0


def test_calculator_forwards_timeout_to_server(fake_root: Path, monkeypatch):
    captured = {}

    class _FakeServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr("rootstock.calculator.RootstockServer", _FakeServer)

    calc = RootstockCalculator(checkpoint=_CHECKPOINT, root=fake_root, device="cpu", timeout=42.0)
    calc._ensure_server()

    assert captured["timeout"] == 42.0


# --- root fallback (parity with the CLI) ------------------------------------


def test_root_falls_back_to_rootstock_root_env(fake_root: Path, monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(fake_root))

    calc = RootstockCalculator(checkpoint=_CHECKPOINT, device="cpu")

    assert calc.root == fake_root


def test_root_falls_back_to_config_file(fake_root: Path, monkeypatch):
    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    from rootstock.config import UserConfig

    monkeypatch.setattr(
        "rootstock.config.load_config", lambda *a, **k: UserConfig(root=str(fake_root))
    )

    calc = RootstockCalculator(checkpoint=_CHECKPOINT, device="cpu")

    assert calc.root == fake_root


def test_env_var_beats_config_file(fake_root: Path, tmp_path_factory, monkeypatch):
    other = tmp_path_factory.mktemp("other")
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(fake_root))
    from rootstock.config import UserConfig

    monkeypatch.setattr("rootstock.config.load_config", lambda *a, **k: UserConfig(root=str(other)))

    calc = RootstockCalculator(checkpoint=_CHECKPOINT, device="cpu")

    assert calc.root == fake_root


def test_no_root_anywhere_raises_with_all_options(monkeypatch):
    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    from rootstock.config import UserConfig

    monkeypatch.setattr("rootstock.config.load_config", lambda *a, **k: UserConfig())

    with pytest.raises(ValueError, match="ROOTSTOCK_ROOT"):
        RootstockCalculator(checkpoint=_CHECKPOINT, device="cpu")


# --- crash recovery ---------------------------------------------------------


@pytest.fixture
def flaky_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Faked install whose calculator kills the worker on the FIRST force call
    only (marker file distinguishes runs) — the GPU-OOM shape, once."""
    import ase

    import rootstock

    pythonpath = os.pathsep.join(
        sorted({str(Path(m.__file__).resolve().parents[1]) for m in (ase, rootstock)})
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setenv("FLAKY_MARKER_FILE", str(tmp_path / "died-once"))

    env_dir = tmp_path / "envs" / "flaky"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(
        f"CHECKPOINTS = {{{_CHECKPOINT!r}: 'dummy'}}\n\n"
        "def setup(checkpoint, device='cpu', **kwargs):\n"
        "    import os\n"
        "    from ase.calculators.lj import LennardJones\n"
        "    class FlakyOnce(LennardJones):\n"
        "        def calculate(self, *a, **k):\n"
        "            marker = os.environ['FLAKY_MARKER_FILE']\n"
        "            if not os.path.exists(marker):\n"
        "                open(marker, 'w').close()\n"
        "                os._exit(9)\n"
        "            return super().calculate(*a, **k)\n"
        "    return FlakyOnce()\n"
    )
    return tmp_path


def test_worker_death_does_not_brick_the_calculator(flaky_env_root: Path):
    atoms = molecule("H2O")
    calc = RootstockCalculator(
        checkpoint=_CHECKPOINT, root=flaky_env_root, device="cpu", timeout=15.0
    )
    try:
        atoms.calc = calc

        with pytest.raises(WorkerDiedError):
            atoms.get_potential_energy()

        # The dead server was torn down, not left as a corpse.
        assert calc._server is None

        # The same calculator instance recovers on the next call.
        energy = atoms.get_potential_energy()
        assert energy == pytest.approx(calc.results["energy"])
    finally:
        calc.close()


def test_close_is_idempotent(fake_root: Path):
    calc = RootstockCalculator(checkpoint=_CHECKPOINT, root=fake_root, device="cpu")
    calc.close()
    calc.close()  # second close must be a no-op, not an error
