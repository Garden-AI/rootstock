"""Cluster-scoped env variants (#208): CLUSTERS parsing and resolution.

An env source may declare ``CLUSTERS = ["polaris"]`` to restrict itself to
some of a shared install's clusters — the mechanism for shipping a
polaris-only variant of an env that declares the same checkpoint ids.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.environment import (
    CheckpointNotFoundError,
    find_env_for_checkpoint,
    parse_clusters_list,
    resolve_checkpoint,
)

UNIVERSAL = """CHECKPOINTS = {"mace-mp-0-medium": "medium", "mace:custom": None}

def setup(checkpoint, device="cuda"):
    return None

def setup_from_path(path, device="cuda"):
    return None
"""

VARIANT = """CLUSTERS = ["polaris"]
CHECKPOINTS = {"mace-mp-0-medium": "medium", "mace:custom": None}

def setup(checkpoint, device="cuda"):
    return None

def setup_from_path(path, device="cuda"):
    return None
"""


def _install(root: Path, name: str, source: str) -> None:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(source)


@pytest.fixture
def shared_root(tmp_path: Path) -> Path:
    _install(tmp_path, "mace", UNIVERSAL)
    _install(tmp_path, "mace-polaris", VARIANT)
    return tmp_path


# --- parse_clusters_list -----------------------------------------------------


def test_parse_clusters_absent_means_universal(tmp_path):
    source = tmp_path / "env.py"
    source.write_text(UNIVERSAL)
    assert parse_clusters_list(source) is None


def test_parse_clusters_list_literal(tmp_path):
    source = tmp_path / "env.py"
    source.write_text('CLUSTERS = ["sophia", "polaris"]\nCHECKPOINTS = {}\n')
    assert parse_clusters_list(source) == ["sophia", "polaris"]


@pytest.mark.parametrize(
    "declaration",
    [
        "CLUSTERS = 'polaris'",  # not a list
        "CLUSTERS = [42]",  # not string literals
        "CLUSTERS = []",  # nobody could serve it
        "CLUSTERS = [name for name in ()]",  # not a literal
    ],
)
def test_parse_clusters_malformed_raises(tmp_path, declaration):
    source = tmp_path / "env.py"
    source.write_text(f"{declaration}\nCHECKPOINTS = {{}}\n")
    with pytest.raises(ValueError):
        parse_clusters_list(source)


# --- resolution: specific beats universal --------------------------------------


def test_variant_wins_on_its_cluster(shared_root):
    env_name, _ = find_env_for_checkpoint(shared_root, "mace-mp-0-medium", "polaris")
    assert env_name == "mace-polaris"


def test_universal_serves_the_other_cluster(shared_root):
    env_name, _ = find_env_for_checkpoint(shared_root, "mace-mp-0-medium", "sophia")
    assert env_name == "mace"


def test_no_cluster_resolves_to_universal(shared_root):
    env_name, _ = find_env_for_checkpoint(shared_root, "mace-mp-0-medium")
    assert env_name == "mace"


def test_custom_ids_follow_the_same_rules(shared_root):
    assert resolve_checkpoint(shared_root, "mace:custom", "polaris").env_name == "mace-polaris"
    assert resolve_checkpoint(shared_root, "mace:custom", "sophia").env_name == "mace"


def test_variant_only_id_without_cluster_asks_for_one(tmp_path):
    _install(tmp_path, "mace-polaris", VARIANT)
    with pytest.raises(CheckpointNotFoundError, match="cluster"):
        find_env_for_checkpoint(tmp_path, "mace-mp-0-medium")


def test_variant_only_id_on_wrong_cluster_errors(tmp_path):
    _install(tmp_path, "mace-polaris", VARIANT)
    with pytest.raises(CheckpointNotFoundError, match="sophia"):
        find_env_for_checkpoint(tmp_path, "mace-mp-0-medium", "sophia")


def test_same_specificity_collision_is_loud(tmp_path):
    # Two unrestricted envs declaring the same id used to resolve by silent
    # directory order; variants make duplicates legitimate, so ambiguity at
    # the same specificity must be an authoring error instead.
    _install(tmp_path, "mace-a", UNIVERSAL)
    _install(tmp_path, "mace-b", UNIVERSAL)
    with pytest.raises(CheckpointNotFoundError, match="several envs"):
        find_env_for_checkpoint(tmp_path, "mace-mp-0-medium")
    with pytest.raises(CheckpointNotFoundError, match="several envs"):
        find_env_for_checkpoint(tmp_path, "mace-mp-0-medium", "sophia")


def test_malformed_clusters_never_widens_a_variant(tmp_path):
    # A broken CLUSTERS drops the env from contention rather than silently
    # serving it everywhere.
    broken = VARIANT.replace('CLUSTERS = ["polaris"]', "CLUSTERS = []")
    _install(tmp_path, "mace", UNIVERSAL)
    _install(tmp_path, "mace-polaris", broken)
    env_name, _ = find_env_for_checkpoint(tmp_path, "mace-mp-0-medium", "polaris")
    assert env_name == "mace"


# --- resolve_current_cluster ----------------------------------------------------


def test_resolve_current_cluster(tmp_path):
    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest
    from rootstock.operations import OperationError, resolve_current_cluster

    cfg = UserConfig(name="t", email="t@t.t")

    solo = tmp_path / "solo"
    save_manifest(create_manifest(solo, ["della"], cfg), solo)
    assert resolve_current_cluster(solo) == "della"
    assert resolve_current_cluster(solo, "della") == "della"

    shared = tmp_path / "shared"
    save_manifest(create_manifest(shared, ["sophia", "polaris"], cfg), shared)
    assert resolve_current_cluster(shared, "polaris") == "polaris"
    with pytest.raises(OperationError, match="sophia, polaris"):
        resolve_current_cluster(shared)  # ambiguous — no honest default
    with pytest.raises(OperationError, match="not one this install serves"):
        resolve_current_cluster(shared, "frontier")  # typo protection

    # No manifest, unregistered root: matches the "unknown" identity a
    # created-on-the-fly manifest would get.
    assert resolve_current_cluster(tmp_path / "fresh") == "unknown"
