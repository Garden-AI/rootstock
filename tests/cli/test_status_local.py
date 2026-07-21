"""Tests for the local-checkpoints section of ``rootstock status``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock import local_checkpoints
from rootstock.commands.status import cmd_status
from rootstock.config import UserConfig
from rootstock.local_checkpoints import (
    record_local_verification,
    register_local_checkpoint,
)
from rootstock.manifest import (
    EnvironmentInfo,
    create_manifest,
    save_manifest,
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
    root = tmp_path / "root"
    env_dir = root / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE)

    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(root, "test", cfg)
    manifest.environments["uma"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00+00:00",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
    )
    save_manifest(manifest, root)
    return root


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


def _make_args(root: Path, **overrides):
    class _Args:
        pass

    args = _Args()
    args.root = str(root)
    args.json = overrides.get("json", False)
    args.sizes = overrides.get("sizes", False)
    return args


def test_status_omits_section_when_no_locals(fake_root, registry, capsys):
    assert cmd_status(_make_args(fake_root)) == 0
    assert "Local checkpoints" not in capsys.readouterr().out


def test_status_shows_verified_local(fake_root, registry, weights, capsys):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    record_local_verification(fake_root, "my-ft", ok=True, device="cuda")

    assert cmd_status(_make_args(fake_root)) == 0
    out = capsys.readouterr().out
    assert "Local checkpoints (this user):" in out
    assert "my-ft" in out
    assert str(weights.resolve()) in out
    assert "✓" in out
    assert "file missing" not in out


def test_status_flags_missing_file_and_unverified(fake_root, registry, weights, capsys):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    weights.unlink()

    cmd_status(_make_args(fake_root))
    out = capsys.readouterr().out
    assert "file missing" in out
    assert "not verified" in out


def test_status_flags_stale_after_env_rebuild(fake_root, registry, weights, capsys):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    # Verified before the (future-dated) env rebuild → stale.
    record_local_verification(fake_root, "my-ft", ok=True, device="cuda")
    from rootstock.manifest import load_manifest

    manifest = load_manifest(fake_root)
    manifest.environments["uma"].built_at = "2999-01-01T00:00:00+00:00"
    save_manifest(manifest, fake_root)

    cmd_status(_make_args(fake_root))
    assert "stale" in capsys.readouterr().out


def test_status_flags_shadowed_local(fake_root, registry, weights, capsys):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    # An env source updated afterwards to declare the same id shadows it.
    (fake_root / "envs" / "uma" / "env_source.py").write_text(
        _ENV_SOURCE.replace('"uma-s-1p1": "uma-s-1p1"', '"my-ft": "x"')
    )
    cmd_status(_make_args(fake_root))
    assert "shadowed" in capsys.readouterr().out


def test_status_json_local_checkpoints(fake_root, registry, weights, capsys):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, setup_kwargs={"task": "omol"})
    record_local_verification(fake_root, "my-ft", ok=True, device="cuda")

    assert cmd_status(_make_args(fake_root, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    entry = payload["local_checkpoints"]["my-ft"]
    assert entry["env"] == "uma"
    assert entry["path"] == str(weights.resolve())
    assert entry["exists"] is True
    assert entry["verified_current"] is True
    assert entry["shadowed_by"] is None
    assert entry["setup_kwargs"] == {"task": "omol"}


def test_status_json_empty_locals_key_present(fake_root, registry, capsys):
    cmd_status(_make_args(fake_root, json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["local_checkpoints"] == {}


def test_status_survives_corrupt_registry(fake_root, registry, capsys):
    registry.write_text("{not json")
    assert cmd_status(_make_args(fake_root)) == 0
    assert "unreadable" in capsys.readouterr().out
