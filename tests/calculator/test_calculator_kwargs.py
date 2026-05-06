"""Tests for RootstockCalculator setup_kwargs validation and env resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.calculator import RootstockCalculator
from rootstock.environment import CheckpointNotFoundError

_MACE_ENV_SOURCE = '''\
"""MACE env."""

CHECKPOINTS = {
    "mace-mp-0-medium": "medium",
}


def setup(checkpoint, device="cuda"):
    return None
'''


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A fake install root with one env (mace) declaring mace-mp-0-medium."""
    env_dir = tmp_path / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_MACE_ENV_SOURCE)
    return tmp_path


def test_setup_kwargs_rejects_reserved_checkpoint(fake_root: Path):
    with pytest.raises(TypeError, match="reserved"):
        RootstockCalculator(
            checkpoint="mace-mp-0-medium",
            root=fake_root,
            setup_kwargs={"checkpoint": "x"},
        )


def test_setup_kwargs_rejects_reserved_device(fake_root: Path):
    with pytest.raises(TypeError, match="reserved"):
        RootstockCalculator(
            checkpoint="mace-mp-0-medium",
            root=fake_root,
            setup_kwargs={"device": "cpu"},
        )


def test_setup_kwargs_rejects_both_reserved(fake_root: Path):
    with pytest.raises(TypeError, match="reserved"):
        RootstockCalculator(
            checkpoint="mace-mp-0-medium",
            root=fake_root,
            setup_kwargs={"checkpoint": "x", "device": "y"},
        )


def test_setup_kwargs_default_is_empty_dict(fake_root: Path):
    calc = RootstockCalculator(checkpoint="mace-mp-0-medium", root=fake_root)
    assert calc.setup_kwargs == {}


def test_setup_kwargs_stored_as_passed(fake_root: Path):
    calc = RootstockCalculator(
        checkpoint="mace-mp-0-medium",
        root=fake_root,
        setup_kwargs={"task": "omol"},
    )
    assert calc.setup_kwargs == {"task": "omol"}


def test_checkpoint_is_required(fake_root: Path):
    with pytest.raises(TypeError):
        RootstockCalculator(root=fake_root)  # missing checkpoint


def test_checkpoint_stored_on_calculator(fake_root: Path):
    calc = RootstockCalculator(checkpoint="mace-mp-0-medium", root=fake_root)
    assert calc.checkpoint == "mace-mp-0-medium"


def test_env_resolved_from_checkpoint(fake_root: Path):
    calc = RootstockCalculator(checkpoint="mace-mp-0-medium", root=fake_root)
    assert calc.env_name == "mace"


def test_unknown_checkpoint_raises(fake_root: Path):
    with pytest.raises(CheckpointNotFoundError):
        RootstockCalculator(checkpoint="not-a-real-id", root=fake_root)
