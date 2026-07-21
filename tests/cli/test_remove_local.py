"""Tests for ``rootstock remove-local``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock import local_checkpoints
from rootstock.commands.local import cmd_remove_local
from rootstock.local_checkpoints import (
    local_checkpoints_for_root,
    register_local_checkpoint,
)

_ENV_SOURCE = """\
CHECKPOINTS = {"uma-s-1p1": "uma-s-1p1"}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    env_dir = tmp_path / "root" / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE)
    return tmp_path / "root"


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "registry.json"
    monkeypatch.setattr(local_checkpoints, "DEFAULT_LOCAL_REGISTRY_FILE", path)
    return path


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "ft.pt"
    path.write_bytes(b"weights")
    return path


def _make_args(root: Path, checkpoint: str):
    class _Args:
        pass

    args = _Args()
    args.checkpoint = checkpoint
    args.root = str(root)
    return args


def test_remove_local_happy_path(fake_root, weights, registry, capsys):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    rc = cmd_remove_local(_make_args(fake_root, "my-ft"))
    assert rc == 0
    assert local_checkpoints_for_root(fake_root) == {}
    assert weights.exists()  # never deletes the weights file
    out = capsys.readouterr().out
    assert "weights file untouched" in out
    assert str(weights.resolve()) in out


def test_remove_local_unknown_id_lists_registered(fake_root, weights, registry, capsys):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    rc = cmd_remove_local(_make_args(fake_root, "typo"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "my-ft" in err
    # Registry untouched by the failed removal.
    assert "my-ft" in local_checkpoints_for_root(fake_root)


def test_remove_local_empty_registry(fake_root, registry, capsys):
    rc = cmd_remove_local(_make_args(fake_root, "my-ft"))
    assert rc == 1
    assert "No local checkpoints" in capsys.readouterr().err
