"""Spawn-time resolution of a checkpoint's weight-prewarm paths (#178).

Tiered: the manifest's exact ``weight_files`` record when one exists (#177),
else a heuristic scan of the shared cache for directories that look like the
env's, else nothing. The manifest read is deliberately raw and best-effort —
it must never print, lock, or refuse a newer schema on the calculator's
spawn path — so corrupt or futuristic manifests just fall back a tier.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock.environment import get_checkpoint_prewarm_paths


def _write_manifest(root: Path, env_name: str, checkpoint: str, weight_files) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "environments": {
                    env_name: {"checkpoints": {checkpoint: {"weight_files": weight_files}}}
                },
            }
        )
    )


def _ship_packages(root: Path, env_name: str, *packages: str) -> None:
    site = root / "envs" / env_name / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for package in packages:
        (site / package).mkdir()


def test_manifest_record_wins(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "uma",
        "uma-s-1p1",
        [
            {"path": "cache/huggingface/hub/models--facebook--UMA/blobs/abc", "size": 7},
            {"path": "home/.cache/fairchem/model.pt", "size": 3},
        ],
    )
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "uma", "uma-s-1p1")
    assert tier == "manifest"
    assert paths == [
        str(tmp_path / "cache/huggingface/hub/models--facebook--UMA/blobs/abc"),
        str(tmp_path / "home/.cache/fairchem/model.pt"),
    ]


def test_manifest_paths_join_split_cache_root(tmp_path: Path):
    """Records are cache-root-relative so they survive split-cache installs."""
    _write_manifest(tmp_path / "install", "uma", "uma-s-1p1", [{"path": "cache/w.pt", "size": 1}])
    paths, tier = get_checkpoint_prewarm_paths(
        tmp_path / "install", "uma", "uma-s-1p1", cache_root=tmp_path / "scratch-cache"
    )
    assert tier == "manifest"
    assert paths == [str(tmp_path / "scratch-cache" / "cache" / "w.pt")]


def test_no_record_falls_back_to_heuristic(tmp_path: Path):
    _ship_packages(tmp_path, "orb", "orb_models", "dgl", "numpy")
    for matching in ("cache/orb", "home/.dgl"):
        (tmp_path / matching).mkdir(parents=True)
    (tmp_path / "cache" / "some-other-env").mkdir()

    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "orb", "orb-v2")
    assert tier == "heuristic"
    # cache/orb matches orb_models at the "_" boundary; home/.dgl matches dgl
    # after the hidden-dir dot; the unrelated dir stays out.
    assert paths == [str(tmp_path / "cache" / "orb"), str(tmp_path / "home" / ".dgl")]


def test_heuristic_matches_env_name_dir(tmp_path: Path):
    _ship_packages(tmp_path, "mace")  # ships nothing that matches by package name
    (tmp_path / "cache" / "mace").mkdir(parents=True)
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "mace", "mace-mp-0-medium")
    assert tier == "heuristic"
    assert paths == [str(tmp_path / "cache" / "mace")]


def test_heuristic_never_includes_shared_hub_trees(tmp_path: Path):
    """huggingface/torch hub trees hold every family's weights — an env that
    ships huggingface_hub or torch must not match the whole shared tree."""
    _ship_packages(tmp_path, "uma", "huggingface_hub", "torch", "fairchem")
    for shared in ("cache/huggingface", "cache/torch"):
        (tmp_path / shared).mkdir(parents=True)
    (tmp_path / "home" / ".cache" / "fairchem").mkdir(parents=True)

    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "uma", "uma-s-1p1")
    assert tier == "heuristic"
    assert paths == [str(tmp_path / "home" / ".cache" / "fairchem")]


def test_heuristic_scans_home_but_skips_dot_cache_itself(tmp_path: Path):
    """home/.cache is scanned per-child; the dir itself must not double in
    as a whole tree (an env shipping a 'cache'-named package would match)."""
    _ship_packages(tmp_path, "weird", "cache")
    (tmp_path / "home" / ".cache" / "unrelated").mkdir(parents=True)
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "weird", "weird-1")
    assert tier == "none"
    assert paths == []


def test_nothing_found_reports_none(tmp_path: Path):
    _ship_packages(tmp_path, "mace", "mace_torch")
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "mace", "mace-mp-0-medium")
    assert (paths, tier) == ([], "none")


def test_custom_checkpoint_skips_heuristic(tmp_path: Path):
    """:custom weights ride checkpoint_path into the prewarm already; with no
    record there is nothing to resolve and nothing worth a 'none' report."""
    _ship_packages(tmp_path, "uma", "fairchem")
    (tmp_path / "home" / ".cache" / "fairchem").mkdir(parents=True)
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "uma", "uma:custom")
    assert (paths, tier) == ([], None)


def test_custom_checkpoint_uses_record_when_present(tmp_path: Path):
    """smoke-test records :custom loads too (#204) — an exact record beats
    skipping even for custom ids."""
    _write_manifest(tmp_path, "uma", "uma:custom", [{"path": "cache/base.pt", "size": 1}])
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "uma", "uma:custom")
    assert tier == "manifest"
    assert paths == [str(tmp_path / "cache" / "base.pt")]


@pytest.mark.parametrize(
    "manifest_body",
    ["not json at all", json.dumps({"environments": "not-a-dict"}), json.dumps([1, 2, 3])],
)
def test_unreadable_manifest_falls_back(tmp_path: Path, manifest_body: str):
    (tmp_path / "manifest.json").write_text(manifest_body)
    _ship_packages(tmp_path, "orb", "orb_models")
    (tmp_path / "cache" / "orb").mkdir(parents=True)
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "orb", "orb-v2")
    assert tier == "heuristic"
    assert paths == [str(tmp_path / "cache" / "orb")]


def test_malformed_record_entries_are_skipped(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "uma",
        "uma-s-1p1",
        ["bare-string", {"size": 5}, {"path": "cache/ok.pt", "size": 5}],
    )
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "uma", "uma-s-1p1")
    assert tier == "manifest"
    assert paths == [str(tmp_path / "cache" / "ok.pt")]


def test_empty_record_falls_back_to_heuristic(tmp_path: Path):
    """#177 never writes empty records (byte floor keeps None instead), so a
    [] here means 'unrecorded', not 'warm nothing'."""
    _write_manifest(tmp_path, "orb", "orb-v2", [])
    _ship_packages(tmp_path, "orb", "orb_models")
    (tmp_path / "cache" / "orb").mkdir(parents=True)
    paths, tier = get_checkpoint_prewarm_paths(tmp_path, "orb", "orb-v2")
    assert tier == "heuristic"
    assert paths == [str(tmp_path / "cache" / "orb")]
