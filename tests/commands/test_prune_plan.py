"""The ``rootstock prune`` planner.

The plan is ``actual − desired``: built envs with no registered source,
checkpoint records no source declares (with their weight files refcounted
against surviving checkpoints), plus the internal-GC tier that needs no
declared state at all. These tests pin the cruft inventory and the safety
constructions from the design (issue #199): refcounting, whole-HF-repo-dir
release, graceful degradation without weight records, and the age guard.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from rootstock.batch import plan_prune
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    compute_source_hash,
    create_manifest,
    save_manifest,
)
from rootstock.operations import OperationError

OLD = "2026-07-01T00:00:00+00:00"
NEWER = "2026-07-02T00:00:00+00:00"
TWO_DAYS_AGO = time.time() - 2 * 24 * 3600


def env_source(*checkpoints: str) -> str:
    entries = "".join(f"    {ckpt!r}: {ckpt!r},\n" for ckpt in checkpoints)
    return (
        f"CHECKPOINTS = {{\n{entries}}}\n\ndef setup(checkpoint, device='cuda'):\n    return None\n"
    )


def register(root: Path, name: str, source: str) -> Path:
    env_file = root / "environments" / f"{name}.py"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(source)
    return env_file


def build(root: Path, name: str, source: str) -> Path:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True, exist_ok=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(source)
    return env_dir


def record(
    root: Path, name: str, *, checkpoints: dict[str, CheckpointInfo] | None = None
) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at=OLD,
        source_hash=compute_source_hash(root / "envs" / name / "env_source.py"),
        source="",
        python_requires=">=3.11",
        dependencies={},
        checkpoints=checkpoints or {},
    )


def save(root: Path, environments: dict[str, EnvironmentInfo]) -> None:
    from rootstock.config import UserConfig

    manifest = create_manifest(root, "test", UserConfig(name="t", email="t@t.t"))
    manifest.environments = environments
    save_manifest(manifest, root)


def weight(root: Path, relpath: str, size: int = 4) -> dict:
    """Write a fake weight file under the cache root; return its record entry."""
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * size)
    return {"path": relpath, "size": size}


def fetched(*weight_entries: dict, recorded: bool = True) -> CheckpointInfo:
    return CheckpointInfo(
        fetched_at=OLD,
        verified_at=NEWER,
        verified_device="cuda",
        weight_files=list(weight_entries) if recorded else None,
    )


def age(path: Path, when: float = TWO_DAYS_AGO) -> None:
    os.utime(path, (when, when))


def checkpoint_items(plan) -> dict[str, object]:
    return {item.checkpoint: item for item in plan.checkpoints}


# -----------------------------------------------------------------------------
# Declarative tier: envs and checkpoints
# -----------------------------------------------------------------------------


def test_converged_root_plans_nothing(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    entry = weight(tmp_path, "cache/mace/model.pt")
    save(
        tmp_path,
        {"mace": record(tmp_path, "mace", checkpoints={"mace-mp-0-medium": fetched(entry)})},
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    assert plan.is_empty
    assert plan.unattributed == []
    assert plan.notes == []


def test_unregistered_built_env_is_pruned_with_its_checkpoints(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    orphan_source = env_source("orb-v2")
    build(tmp_path, "orb", orphan_source)  # built, never registered
    entry = weight(tmp_path, "cache/orb/orb-v2.ckpt")
    save(
        tmp_path,
        {
            "mace": record(tmp_path, "mace"),
            "orb": record(tmp_path, "orb", checkpoints={"orb-v2": fetched(entry)}),
        },
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    assert [(e.env_name, e.reason) for e in plan.envs] == [("orb", "no registered source")]
    assert plan.envs[0].env_dir and plan.envs[0].source_file is None
    item = checkpoint_items(plan)["orb-v2"]
    assert item.reason == "env pruned"
    assert item.files == ["cache/orb/orb-v2.ckpt"]
    assert item.reclaim_bytes == 4


def test_undeclared_checkpoint_refcounts_against_survivors(tmp_path):
    source = env_source("mace-mp-0-medium")  # small was dropped from CHECKPOINTS
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    shared = weight(tmp_path, "cache/mace/shared.pt")
    solo = weight(tmp_path, "cache/mace/small-only.pt")
    save(
        tmp_path,
        {
            "mace": record(
                tmp_path,
                "mace",
                checkpoints={
                    "mace-mp-0-medium": fetched(shared),
                    "mace-mp-0-small": fetched(shared, solo),
                },
            )
        },
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    assert plan.envs == []
    item = checkpoint_items(plan)["mace-mp-0-small"]
    assert item.reason == "not declared"
    assert item.files == ["cache/mace/small-only.pt"]  # shared.pt survives with medium


def test_wholly_released_hf_repo_dir_is_removed_as_a_unit(tmp_path):
    source = env_source("uma-s-1p1")  # uma-m was dropped
    register(tmp_path, "uma", source)
    build(tmp_path, "uma", source)
    released = weight(tmp_path, "cache/huggingface/hub/models--facebook--UMA-m/blobs/aa")
    weight(
        tmp_path, "cache/huggingface/hub/models--facebook--UMA-m/refs/main"
    )  # unrecorded residue
    kept_blob = weight(tmp_path, "cache/huggingface/hub/models--facebook--UMA/blobs/bb")
    save(
        tmp_path,
        {
            "uma": record(
                tmp_path,
                "uma",
                checkpoints={
                    "uma-s-1p1": fetched(kept_blob),
                    "uma-m": fetched(released, kept_blob),
                },
            )
        },
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    # The released repo dir goes whole (covering the unrecorded refs/ file);
    # the shared repo survives because uma-s-1p1 still records into it.
    assert [d.path for d in plan.weight_dirs] == ["cache/huggingface/hub/models--facebook--UMA-m"]
    assert plan.weight_dirs[0].reclaim_bytes == 8
    item = checkpoint_items(plan)["uma-m"]
    assert item.files == []  # nothing left outside the dir item / survivors


def test_unrecorded_checkpoint_degrades_to_record_drop(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    weight(tmp_path, "cache/mace/mystery.pt")  # on disk, but no record attributes it
    save(
        tmp_path,
        {
            "mace": record(
                tmp_path,
                "mace",
                checkpoints={
                    "mace-mp-0-medium": fetched(recorded=True),
                    "mace-off23-small": fetched(recorded=False),
                },
            )
        },
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    item = checkpoint_items(plan)["mace-off23-small"]
    assert item.recorded is False
    assert item.files == [] and item.reclaim_bytes == 0
    # Its weights (which nothing attributes) surface in the tier-2 report.
    assert [Path(u["path"]).name for u in plan.unattributed] == ["mace"]


def test_hand_deleted_weights_prune_the_stale_record(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    entry = {"path": "cache/mace/model.pt", "size": 4}  # recorded but never written
    save(
        tmp_path,
        {"mace": record(tmp_path, "mace", checkpoints={"mace-mp-0-medium": fetched(entry)})},
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    item = checkpoint_items(plan)["mace-mp-0-medium"]
    assert item.reason == "weights missing on disk"
    assert item.files == []


def test_source_dir_mode_unregisters_beyond_the_declared_set(tmp_path):
    for name in ("mace", "uma"):
        source = env_source(f"{name}-ckpt")
        register(tmp_path, name, source)
        build(tmp_path, name, source)
        (tmp_path / "environments" / f"{name}.py.lock").write_text("lock")
    save(tmp_path, {"mace": record(tmp_path, "mace"), "uma": record(tmp_path, "uma")})
    declared = tmp_path / "declared"
    declared.mkdir()
    (declared / "mace.py").write_text(env_source("mace-ckpt"))

    plan = plan_prune(tmp_path, source_dir=declared, cache_root=tmp_path)

    assert [(e.env_name, e.env_dir) for e in plan.envs] == [("uma", True)]
    assert plan.envs[0].source_file == str(tmp_path / "environments" / "uma.py")
    assert plan.envs[0].reason == f"not in {declared}"


def test_empty_source_dir_is_an_error(tmp_path):
    empty = tmp_path / "declared"
    empty.mkdir()
    with pytest.raises(OperationError, match="No \\*.py environment files"):
        plan_prune(tmp_path, source_dir=empty, cache_root=tmp_path)


def test_unparseable_checkpoints_keeps_records(tmp_path):
    source = "CHECKPOINTS = 'not a dict'\n"
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    save(
        tmp_path,
        {"mace": record(tmp_path, "mace", checkpoints={"mace-mp-0-medium": fetched()})},
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    # Sync skips *adding* on a parse error; prune must skip *removing*.
    assert plan.checkpoints == []
    assert any("keeping all of its checkpoint records" in note for note in plan.notes)


def test_missing_environments_dir_gets_a_loud_note(tmp_path):
    build(tmp_path, "mace", env_source("mace-mp-0-medium"))
    save(tmp_path, {"mace": record(tmp_path, "mace")})

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    assert [e.env_name for e in plan.envs] == ["mace"]
    assert any("every built env counts as undeclared" in note for note in plan.notes)


def test_suspicious_recorded_paths_are_refused(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    save(
        tmp_path,
        {
            "mace": record(
                tmp_path,
                "mace",
                checkpoints={
                    "gone": fetched(
                        {"path": "../outside.pt", "size": 4},
                        {"path": "/etc/passwd", "size": 4},
                    )
                },
            )
        },
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    item = checkpoint_items(plan)["gone"]
    assert item.files == []
    assert sum("refusing suspicious recorded path" in note for note in plan.notes) == 2


def test_env_and_checkpoint_filters_limit_scope(tmp_path):
    register(tmp_path, "mace", env_source("mace-mp-0-medium"))
    build(tmp_path, "orb", env_source("orb-v2"))
    build(tmp_path, "gemnet", env_source("g1"))
    save(
        tmp_path,
        {
            "orb": record(tmp_path, "orb", checkpoints={"orb-v2": fetched()}),
            "gemnet": record(tmp_path, "gemnet", checkpoints={"g1": fetched()}),
        },
    )

    plan = plan_prune(tmp_path, envs=["orb"], cache_root=tmp_path)
    assert [e.env_name for e in plan.envs] == ["orb"]
    assert [c.checkpoint for c in plan.checkpoints] == ["orb-v2"]

    plan = plan_prune(tmp_path, checkpoints=["g1"], cache_root=tmp_path)
    assert [c.checkpoint for c in plan.checkpoints] == ["g1"]

    with pytest.raises(OperationError, match="Unknown environment"):
        plan_prune(tmp_path, envs=["nope"], cache_root=tmp_path)
    with pytest.raises(OperationError, match="Unknown checkpoint id"):
        plan_prune(tmp_path, checkpoints=["nope"], cache_root=tmp_path)


# -----------------------------------------------------------------------------
# Internal GC tier
# -----------------------------------------------------------------------------


def test_stale_build_and_trash_entries_are_collected(tmp_path):
    stale_build = tmp_path / ".build" / "mace.12345"
    stale_build.mkdir(parents=True)
    age(stale_build)
    fresh_build = tmp_path / ".build" / "uma.99999"
    fresh_build.mkdir()
    stale_trash = tmp_path / ".trash" / "orb.1720000000"
    stale_trash.mkdir(parents=True)
    age(stale_trash)

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    collected = {(item.kind, Path(item.path).name) for item in plan.gc}
    assert ("build", "mace.12345") in collected
    assert ("trash", "orb.1720000000") in collected
    assert ("build", "uma.99999") not in collected  # age guard
    assert any("younger than --min-age" in note for note in plan.notes)


def test_orphaned_interpreters_are_collected_live_ones_kept(tmp_path):
    live = tmp_path / ".python" / "cpython-3.11.9-linux-x86_64-gnu"
    (live / "bin").mkdir(parents=True)
    (live / "bin" / "python3.11").touch()
    orphan = tmp_path / ".python" / "cpython-3.10.14-linux-x86_64-gnu"
    orphan.mkdir()
    age(orphan)
    stranded = tmp_path / ".python" / "cpython-3.12.1.installing.4242.1"
    stranded.mkdir()
    age(stranded)
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    env_dir = tmp_path / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(live / "bin" / "python3.11")
    (env_dir / "env_source.py").write_text(source)
    save(tmp_path, {"mace": record(tmp_path, "mace", checkpoints={"mace-mp-0-medium": fetched()})})

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    by_name = {Path(item.path).name: item for item in plan.gc if item.kind == "interpreter"}
    assert set(by_name) == {orphan.name, stranded.name}
    assert by_name[stranded.name].reason == "stranded staging copy"


def test_orphaned_lockfile_is_collected(tmp_path):
    register(tmp_path, "mace", env_source("mace-mp-0-medium"))
    build(tmp_path, "mace", env_source("mace-mp-0-medium"))
    (tmp_path / "environments" / "mace.py.lock").write_text("lock")  # has a source: kept
    orphan = tmp_path / "environments" / "deleted.py.lock"
    orphan.write_text("lock")
    age(orphan)
    save(tmp_path, {"mace": record(tmp_path, "mace", checkpoints={"mace-mp-0-medium": fetched()})})

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    assert [Path(i.path).name for i in plan.gc if i.kind == "lockfile"] == ["deleted.py.lock"]


def test_uv_cache_delegates_to_uv(tmp_path):
    (tmp_path / ".uv-cache").mkdir()

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    items = [i for i in plan.gc if i.kind == "uv-cache"]
    assert len(items) == 1
    assert items[0].reclaim_bytes is None  # unknown until uv runs


def test_gc_only_never_touches_envs_or_checkpoints(tmp_path):
    build(tmp_path, "orb", env_source("orb-v2"))  # unregistered: declarative cruft
    save(tmp_path, {"orb": record(tmp_path, "orb", checkpoints={"orb-v2": fetched()})})
    stale = tmp_path / ".build" / "orb.1"
    stale.mkdir(parents=True)
    age(stale)

    plan = plan_prune(tmp_path, gc_only=True, cache_root=tmp_path)

    assert plan.envs == [] and plan.checkpoints == [] and plan.unattributed == []
    assert [i.kind for i in plan.gc] == ["build"]


# -----------------------------------------------------------------------------
# Tier 2: unattributed cache
# -----------------------------------------------------------------------------


def test_unattributed_cache_is_reported_not_deleted(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    entry = weight(tmp_path, "cache/mace/model.pt")
    weight(tmp_path, "cache/mystery/leftover.bin", size=10)
    weight(tmp_path, "cache/huggingface/hub/models--never--read/blobs/cc", size=7)
    weight(tmp_path, "home/.cache/matgl/thing.pt", size=5)
    save(
        tmp_path,
        {"mace": record(tmp_path, "mace", checkpoints={"mace-mp-0-medium": fetched(entry)})},
    )

    plan = plan_prune(tmp_path, cache_root=tmp_path)

    reported = {Path(u["path"]).name: u["bytes"] for u in plan.unattributed}
    assert reported == {"mystery": 10, "models--never--read": 7, "matgl": 5}
    assert plan.is_empty  # report-only: nothing to execute

    deep = plan_prune(tmp_path, deep=True, cache_root=tmp_path)
    assert deep.unattributed == []
    assert {Path(i.path).name for i in deep.gc if i.kind == "unattributed"} == {
        "mystery",
        "models--never--read",
        "matgl",
    }


# -----------------------------------------------------------------------------
# Progress
# -----------------------------------------------------------------------------


def test_planner_emits_progress_before_slow_tree_walks(tmp_path):
    """Planning walks whole trees for byte counts; someone tailing a batch
    job's outfile must see each walk announced before it starts."""
    build(tmp_path, "orb", env_source("orb-v2"))  # unregistered: gets sized
    save(tmp_path, {"orb": record(tmp_path, "orb")})
    stale = tmp_path / ".build" / "orb.1"
    stale.mkdir(parents=True)
    age(stale)
    weight(tmp_path, "cache/mystery/junk.bin")

    lines: list[str] = []
    plan_prune(tmp_path, cache_root=tmp_path, progress=lines.append)

    assert any(line.startswith("reading install state") for line in lines)
    assert "sizing envs/orb" in lines
    assert any(line == f"sizing {stale}" for line in lines)
    assert "scanning cache/home for unattributed contents" in lines
    assert "sizing cache/mystery" in lines
