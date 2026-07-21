"""Tests for resolve_checkpoint (canonical + local overlay)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.environment import CheckpointNotFoundError
from rootstock.local_checkpoints import (
    register_local_checkpoint,
    resolve_checkpoint,
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
    return root


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "ft.pt"
    path.write_bytes(b"weights")
    return path


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    return tmp_path / "registry.json"


def test_canonical_hit(fake_root, registry):
    resolved = resolve_checkpoint(fake_root, "uma-s-1p1", registry_path=registry)
    assert resolved.env_name == "uma"
    assert resolved.path is None
    assert resolved.setup_kwargs == {}
    assert not resolved.is_local


def test_local_hit(fake_root, weights, registry):
    register_local_checkpoint(
        fake_root,
        "my-ft",
        "uma",
        weights,
        setup_kwargs={"task": "omol"},
        registry_path=registry,
    )
    resolved = resolve_checkpoint(fake_root, "my-ft", registry_path=registry)
    assert resolved.is_local
    assert resolved.env_name == "uma"
    assert resolved.path == str(weights.resolve())
    assert resolved.setup_kwargs == {"task": "omol"}


def test_canonical_shadows_local(fake_root, weights, registry):
    # Registration prevents this direction, but an env installed *after* a
    # local registration can introduce a collision — canonical wins.
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    (fake_root / "envs" / "uma" / "env_source.py").write_text(
        _ENV_SOURCE.replace('"uma-s-1p1": "uma-s-1p1"', '"my-ft": "x", "uma-s-1p1": "y"')
    )
    resolved = resolve_checkpoint(fake_root, "my-ft", registry_path=registry)
    assert not resolved.is_local


def test_miss_mentions_both_namespaces(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    with pytest.raises(CheckpointNotFoundError) as exc:
        resolve_checkpoint(fake_root, "typo", registry_path=registry)
    msg = str(exc.value)
    assert "uma-s-1p1" in msg  # canonical listing preserved
    assert "my-ft" in msg  # registered local ids listed
    assert "add-local" in msg  # registration hint


def test_miss_without_locals_still_hints_add_local(fake_root, registry):
    with pytest.raises(CheckpointNotFoundError, match="add-local"):
        resolve_checkpoint(fake_root, "typo", registry_path=registry)


def test_corrupt_registry_does_not_affect_canonical(fake_root, registry):
    # Canonical ids resolve before the registry is consulted at all.
    registry.write_text("{not json")
    resolved = resolve_checkpoint(fake_root, "uma-s-1p1", registry_path=registry)
    assert resolved.env_name == "uma"


def test_corrupt_registry_warns_and_misses_cleanly(fake_root, registry, capsys):
    # A broken per-user file must surface as CheckpointNotFoundError (with a
    # warning), never as a LocalCheckpointError crash.
    registry.write_text("{not json")
    with pytest.raises(CheckpointNotFoundError):
        resolve_checkpoint(fake_root, "typo", registry_path=registry)
    assert "ignoring local-checkpoint registry" in capsys.readouterr().err
