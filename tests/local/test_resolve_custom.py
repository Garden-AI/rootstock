"""Tests for '<family>:custom' CHECKPOINTS entries: resolution, the parse-time
split that keeps them out of the canonical dict, and the colon lint."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.environment import (
    CheckpointNotFoundError,
    parse_checkpoints_dict,
    parse_custom_checkpoint_ids,
    resolve_checkpoint,
)

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

_MULTI_FAMILY_ENV_SOURCE = """\
CHECKPOINTS = {
    "esen-sm-direct-all-omol": "esen-sm-direct-all-omol",
    "allscaip-md-conserving-all-omol": "allscaip-md-conserving-all-omol",
    "esen:custom": None,
    "allscaip:custom": None,
}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""

_NO_CUSTOM_ENV_SOURCE = """\
CHECKPOINTS = {"orb-v2": "orb-v2"}


def setup(checkpoint, device="cuda"):
    return None
"""


def _make_env(root: Path, name: str, source: str) -> None:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(source)


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    _make_env(root, "uma", _ENV_SOURCE)
    return root


# ---------- resolution ---------------------------------------------------------


def test_custom_entry_resolves_to_declaring_env(fake_root):
    resolved = resolve_checkpoint(fake_root, "uma:custom")
    assert resolved.env_name == "uma"
    assert resolved.is_custom
    assert resolved.checkpoint == "uma:custom"


def test_multiple_families_resolve_to_the_same_env(tmp_path):
    # The key's family prefix is naming, not mechanism: which env hosts the
    # id is determined by which env's dict declares it, so a multi-family
    # env declares one entry per user-facing family.
    root = tmp_path / "root"
    _make_env(root, "fairchem_v2", _MULTI_FAMILY_ENV_SOURCE)
    for checkpoint in ("esen:custom", "allscaip:custom"):
        resolved = resolve_checkpoint(root, checkpoint)
        assert resolved.env_name == "fairchem_v2"
        assert resolved.is_custom


def test_canonical_id_resolves_without_custom_flag(fake_root):
    resolved = resolve_checkpoint(fake_root, "uma-s-1p1")
    assert resolved.env_name == "uma"
    assert not resolved.is_custom


def test_unknown_custom_id_lists_declared_entries(fake_root):
    with pytest.raises(CheckpointNotFoundError) as exc:
        resolve_checkpoint(fake_root, "umma:custom")
    msg = str(exc.value)
    assert "umma:custom" in msg
    assert "uma:custom" in msg  # the declared entries are the menu


def test_no_custom_entries_anywhere_points_at_maintainer(tmp_path):
    root = tmp_path / "root"
    _make_env(root, "orb", _NO_CUSTOM_ENV_SOURCE)
    with pytest.raises(CheckpointNotFoundError, match="maintainer"):
        resolve_checkpoint(root, "orb:custom")


def test_unknown_canonical_id_still_lists_canonicals(fake_root):
    with pytest.raises(CheckpointNotFoundError, match="uma-s-1p1"):
        resolve_checkpoint(fake_root, "uma-s-9p9")


# ---------- the parse-time split ----------------------------------------------


def test_parse_strips_custom_entries_from_canonical_dict(tmp_path):
    # add/smoke-test/status iterate the canonical dict; the sentinel must
    # never leak into it.
    source = tmp_path / "env.py"
    source.write_text(_ENV_SOURCE)
    assert parse_checkpoints_dict(source) == {"uma-s-1p1": "uma-s-1p1"}
    assert parse_custom_checkpoint_ids(source) == ["uma:custom"]


def test_parse_custom_ids_empty_without_entries(tmp_path):
    source = tmp_path / "env.py"
    source.write_text(_NO_CUSTOM_ENV_SOURCE)
    assert parse_custom_checkpoint_ids(source) == []


# ---------- colon lint ---------------------------------------------------------


def _write_env(tmp_path: Path, checkpoints_literal: str) -> Path:
    source = tmp_path / "env.py"
    source.write_text(
        f"CHECKPOINTS = {checkpoints_literal}\n\ndef setup(c, device='cuda'):\n    return None\n"
    )
    return source


@pytest.mark.parametrize(
    "key",
    ["bad:id", ":custom", "a:b:custom", "custom:uma-s-1p1"],
    ids=["arbitrary-colon", "empty-family", "nested-colon", "old-prefix-form"],
)
def test_colon_keys_other_than_family_custom_rejected(tmp_path, key):
    source = _write_env(tmp_path, f'{{"{key}": "x"}}')
    with pytest.raises(ValueError, match="reserved"):
        parse_checkpoints_dict(source)


def test_custom_entry_value_must_be_none(tmp_path):
    source = _write_env(tmp_path, '{"uma:custom": "uma-s-1p1"}')
    with pytest.raises(ValueError, match="None"):
        parse_checkpoints_dict(source)


def test_canonical_value_may_not_be_none(tmp_path):
    source = _write_env(tmp_path, '{"uma-s-1p1": None}')
    with pytest.raises(ValueError, match="string literals"):
        parse_checkpoints_dict(source)
