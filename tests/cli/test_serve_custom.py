"""Tests for ``rootstock serve`` guards on ':custom' checkpoints (--weights)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock import cli
from rootstock.commands.serve import cmd_serve

_ENV_SOURCE = """\
CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma:custom": None,
}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""

_ENTRY_NO_HOOK_ENV_SOURCE = """\
CHECKPOINTS = {
    "orb-v2": "orb-v2",
    "orb:custom": None,
}


def setup(checkpoint, device="cuda"):
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
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "ft.pt"
    path.write_bytes(b"weights")
    return path


def _make_args(root: Path, checkpoint: str, **overrides):
    class _Args:
        pass

    args = _Args()
    args.root = str(root)
    args.checkpoint = checkpoint
    args.socket = str(root / "sock")
    args.device = overrides.get("device", "cpu")
    args.kwarg = overrides.get("kwarg")
    args.weights = overrides.get("weights")
    return args


def test_parser_accepts_weights_flag(monkeypatch):
    seen = {}

    def fake_serve(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "cmd_serve", fake_serve)
    monkeypatch.setattr(
        "sys.argv",
        ["rootstock", "serve", "uma:custom", "--socket", "/tmp/s", "--weights", "/x/ft.pt"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    assert seen["args"].weights == "/x/ft.pt"


def test_serve_custom_without_weights(fake_root, capsys):
    rc = cmd_serve(_make_args(fake_root, "uma:custom"))
    assert rc == 2
    assert "--weights" in capsys.readouterr().err


def test_serve_weights_without_custom_names_the_entry(fake_root, weights, capsys):
    rc = cmd_serve(_make_args(fake_root, "uma-s-1p1", weights=str(weights)))
    assert rc == 2
    assert "uma:custom" in capsys.readouterr().err


def test_serve_custom_missing_weights_file(fake_root, weights, capsys):
    weights.unlink()
    rc = cmd_serve(_make_args(fake_root, "uma:custom", weights=str(weights)))
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_serve_custom_hookless_env(tmp_path, weights, capsys):
    env_dir = tmp_path / "root" / "envs" / "orb"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENTRY_NO_HOOK_ENV_SOURCE)
    rc = cmd_serve(_make_args(tmp_path / "root", "orb:custom", weights=str(weights)))
    assert rc == 2
    assert "maintainer" in capsys.readouterr().err


def test_serve_custom_rejects_path_kwarg(fake_root, weights, capsys):
    rc = cmd_serve(_make_args(fake_root, "uma:custom", weights=str(weights), kwarg=["path=/x"]))
    assert rc == 2
    assert "path" in capsys.readouterr().err
