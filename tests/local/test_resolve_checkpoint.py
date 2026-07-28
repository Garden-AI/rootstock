"""Tests for resolve_checkpoint's canonical-id path (':custom' entries are
covered in test_resolve_custom.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.environment import CheckpointNotFoundError, resolve_checkpoint

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


def test_canonical_hit(fake_root):
    resolved = resolve_checkpoint(fake_root, "uma-s-1p1")
    assert resolved.env_name == "uma"
    assert not resolved.is_custom
    assert resolved.checkpoint == "uma-s-1p1"


def test_miss_lists_canonical_ids(fake_root):
    with pytest.raises(CheckpointNotFoundError) as exc:
        resolve_checkpoint(fake_root, "typo")
    msg = str(exc.value)
    assert "uma-s-1p1" in msg  # canonical listing preserved
    assert "rootstock install" in msg  # hint for not-yet-installed envs


def test_miss_lists_custom_entries_when_declared(tmp_path):
    # A declared ':custom' entry is part of the menu: the not-found listing
    # shows it verbatim, plus the weights= pattern line, once.
    root = tmp_path / "root"
    env_dir = root / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    with_entry = _ENV_SOURCE.replace(
        '"uma-s-1p1": "uma-s-1p1"', '"uma-s-1p1": "uma-s-1p1", "uma:custom": None'
    )
    (env_dir / "env_source.py").write_text(with_entry)
    with pytest.raises(CheckpointNotFoundError) as exc:
        resolve_checkpoint(root, "typo")
    msg = str(exc.value)
    assert "uma:custom" in msg
    assert "weights" in msg


def test_miss_omits_custom_hint_without_entries(fake_root):
    # The fixture env declares the hook but no ':custom' entry (a built env
    # predating the entries): user weights are unavailable, so nothing may
    # be advertised.
    with pytest.raises(CheckpointNotFoundError) as exc:
        resolve_checkpoint(fake_root, "typo")
    assert ":custom" not in str(exc.value)


def test_miss_on_empty_root(tmp_path):
    with pytest.raises(CheckpointNotFoundError, match="No envs are installed"):
        resolve_checkpoint(tmp_path / "root", "uma-s-1p1")
