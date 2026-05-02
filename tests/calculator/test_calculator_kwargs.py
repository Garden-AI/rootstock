"""Tests for RootstockCalculator setup_kwargs validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.calculator import RootstockCalculator


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    # Create a fake "built" env so the constructor's existence check passes.
    env_python = tmp_path / "envs" / "mace_env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.touch()
    return tmp_path


def test_setup_kwargs_rejects_reserved_model(fake_root: Path):
    with pytest.raises(TypeError, match="reserved"):
        RootstockCalculator(
            model="mace",
            checkpoint="medium",
            root=fake_root,
            setup_kwargs={"model": "x"},
        )


def test_setup_kwargs_rejects_reserved_device(fake_root: Path):
    with pytest.raises(TypeError, match="reserved"):
        RootstockCalculator(
            model="mace",
            checkpoint="medium",
            root=fake_root,
            setup_kwargs={"device": "cpu"},
        )


def test_setup_kwargs_rejects_both_reserved(fake_root: Path):
    with pytest.raises(TypeError, match="reserved"):
        RootstockCalculator(
            model="mace",
            checkpoint="medium",
            root=fake_root,
            setup_kwargs={"model": "x", "device": "y"},
        )


def test_setup_kwargs_default_is_empty_dict(fake_root: Path):
    calc = RootstockCalculator(model="mace", checkpoint="medium", root=fake_root)
    assert calc.setup_kwargs == {}


def test_setup_kwargs_stored_as_passed(fake_root: Path):
    calc = RootstockCalculator(
        model="mace",
        checkpoint="medium",
        root=fake_root,
        setup_kwargs={"task": "omol"},
    )
    assert calc.setup_kwargs == {"task": "omol"}


def test_checkpoint_is_required(fake_root: Path):
    with pytest.raises(TypeError):
        RootstockCalculator(model="mace", root=fake_root)  # missing checkpoint


def test_checkpoint_stored_on_calculator(fake_root: Path):
    calc = RootstockCalculator(model="mace", checkpoint="medium", root=fake_root)
    assert calc.model_arg == "medium"
