"""Tests for ``rootstock serve`` guards on local (user-registered) checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock import local_checkpoints
from rootstock.commands.serve import cmd_serve
from rootstock.local_checkpoints import register_local_checkpoint

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


@pytest.fixture
def registered(fake_root, weights, registry) -> str:
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    return "my-ft"


def _make_args(root: Path, checkpoint: str, **overrides):
    class _Args:
        pass

    args = _Args()
    args.root = str(root)
    args.checkpoint = checkpoint
    args.socket = str(root / "sock")
    args.device = overrides.get("device", "cpu")
    args.kwarg = overrides.get("kwarg")
    return args


def test_serve_rejects_path_kwarg_for_local(fake_root, registered, capsys):
    rc = cmd_serve(_make_args(fake_root, registered, kwarg=["path=/x"]))
    assert rc == 2
    assert "reserved" in capsys.readouterr().err


def test_serve_missing_weights_file(fake_root, registered, weights, capsys):
    weights.unlink()
    rc = cmd_serve(_make_args(fake_root, registered))
    assert rc == 1
    assert "no longer exists" in capsys.readouterr().err
