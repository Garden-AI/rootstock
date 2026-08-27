"""Checkpoint -> batched-env resolution."""

import pytest

from rootstock.environment import CheckpointNotFoundError, find_batched_env_for_checkpoint

ASE_SOURCE = """
CHECKPOINTS = {"mace-mp-0-medium": "medium"}


def setup(checkpoint, device="cuda"):
    pass
"""

BATCHED_SOURCE = """
CHECKPOINTS = {"mace-mp-0-medium": "medium"}


def setup_batched(checkpoint, device="cuda"):
    pass
"""


def _make_env(root, name, source):
    env_dir = root / "envs" / name
    env_dir.mkdir(parents=True)
    (env_dir / "env_source.py").write_text(source)


def test_same_id_resolves_to_batched_env(tmp_path):
    # The same canonical id declared by an ASE-only env and a batched-capable
    # env is not ambiguous: capability filtering picks the batched one.
    _make_env(tmp_path, "mace", ASE_SOURCE)
    _make_env(tmp_path, "mace_nvalchemi", BATCHED_SOURCE)
    assert find_batched_env_for_checkpoint(tmp_path, "mace-mp-0-medium") == "mace_nvalchemi"


def test_unknown_id_lists_batched_menu(tmp_path):
    _make_env(tmp_path, "mace_nvalchemi", BATCHED_SOURCE)
    with pytest.raises(CheckpointNotFoundError, match="mace-mp-0-medium"):
        find_batched_env_for_checkpoint(tmp_path, "nope")


def test_ase_only_install_points_at_missing_capability(tmp_path):
    _make_env(tmp_path, "mace", ASE_SOURCE)
    with pytest.raises(CheckpointNotFoundError, match="setup_batched"):
        find_batched_env_for_checkpoint(tmp_path, "mace-mp-0-medium")


def test_empty_install(tmp_path):
    with pytest.raises(CheckpointNotFoundError, match="No envs are installed"):
        find_batched_env_for_checkpoint(tmp_path, "mace-mp-0-medium")
