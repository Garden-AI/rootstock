"""The ``rootstock prune`` executor.

What's under test is the safety contract: checkpoint records drop (one
manifest transaction) *before* any weight file is unlinked, envs disappear
through a trash-rename, failures keep going without sinking the batch, and
exactly one manifest refresh runs at the end when anything succeeded.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rootstock.batch import (
    GCItem,
    PruneCheckpointItem,
    PruneEnvItem,
    PrunePlan,
    PruneWeightDirItem,
    execute_prune,
)
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    create_manifest,
    load_manifest,
    save_manifest,
)


@pytest.fixture
def refreshes(monkeypatch) -> list:
    recorded: list = []
    monkeypatch.setattr(
        "rootstock.operations.update_and_push_manifest",
        lambda root, **kwargs: recorded.append(kwargs) or True,
    )
    return recorded


def save(root: Path, environments: dict[str, EnvironmentInfo]) -> None:
    from rootstock.config import UserConfig

    manifest = create_manifest(root, "test", UserConfig(name="t", email="t@t.t"))
    manifest.environments = environments
    save_manifest(manifest, root)


def env_record(checkpoints: dict[str, CheckpointInfo] | None = None) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at="2026-07-01T00:00:00+00:00",
        source_hash=None,
        source="",
        python_requires=">=3.11",
        dependencies={},
        checkpoints=checkpoints or {},
    )


def touch(root: Path, relpath: str, size: int = 4) -> Path:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * size)
    return target


def statuses(report) -> dict[tuple[str, str], str]:
    return {(r.phase, r.label): r.status for r in report.results}


def test_happy_path_drops_records_deletes_files_and_refreshes_once(tmp_path, refreshes):
    save(
        tmp_path,
        {
            "mace": env_record(
                {"keep": CheckpointInfo(fetched_at="x"), "gone": CheckpointInfo(fetched_at="x")}
            )
        },
    )
    touch(tmp_path, "cache/mace/deep/nested/gone.pt")
    touch(tmp_path, "cache/mace/keep.pt")
    plan = PrunePlan(
        checkpoints=[
            PruneCheckpointItem(
                "mace", "gone", "not declared", files=["cache/mace/deep/nested/gone.pt"]
            )
        ]
    )

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    assert statuses(report) == {("checkpoint", "mace/gone"): "ok"}
    manifest = load_manifest(tmp_path)
    assert set(manifest.environments["mace"].checkpoints) == {"keep"}
    assert not (tmp_path / "cache/mace/deep").exists()  # emptied parents pruned
    assert (tmp_path / "cache/mace/keep.pt").exists()  # stops at first non-empty dir
    assert (tmp_path / "cache").exists()
    assert len(refreshes) == 1 and refreshes[0] == {"quiet": True, "push": True}


def test_weight_dir_item_removes_the_whole_repo_dir(tmp_path, refreshes):
    touch(tmp_path, "cache/huggingface/hub/models--a--b/blobs/aa")
    touch(tmp_path, "cache/huggingface/hub/models--a--b/refs/main")
    touch(tmp_path, "cache/huggingface/hub/models--keep--me/blobs/bb")
    plan = PrunePlan(
        weight_dirs=[
            PruneWeightDirItem("cache/huggingface/hub/models--a--b", "released", ["uma-m"], 8)
        ]
    )

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    assert statuses(report) == {("weights", "models--a--b"): "ok"}
    assert not (tmp_path / "cache/huggingface/hub/models--a--b").exists()
    assert (tmp_path / "cache/huggingface/hub/models--keep--me/blobs/bb").exists()


def test_env_removal_goes_through_trash_rename(tmp_path, refreshes, monkeypatch):
    env_dir = tmp_path / "envs" / "orb"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    source = touch(tmp_path, "environments/orb.py")
    touch(tmp_path, "environments/orb.py.lock")
    save(tmp_path, {"orb": env_record()})

    removed: list[Path] = []
    real_rmtree = shutil.rmtree

    def spying_rmtree(path, *args, **kwargs):
        removed.append(Path(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("rootstock.batch.shutil.rmtree", spying_rmtree)
    plan = PrunePlan(
        envs=[PruneEnvItem("orb", "not declared", env_dir=True, source_file=str(source))]
    )

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    assert statuses(report) == {("env", "orb"): "ok"}
    assert not env_dir.exists()
    # The rmtree target was the .trash rename, never the live envs/ path —
    # a mid-spawn reader sees atomic disappearance, not a half-deleted tree.
    assert len(removed) == 1
    assert removed[0].parent == tmp_path / ".trash"
    assert removed[0].name.startswith("orb.")
    assert not source.exists()
    assert not (tmp_path / "environments/orb.py.lock").exists()


def test_record_drop_failure_blocks_weights_but_not_envs(tmp_path, refreshes, monkeypatch):
    save(tmp_path, {"mace": env_record({"gone": CheckpointInfo(fetched_at="x")})})
    weight_file = touch(tmp_path, "cache/mace/gone.pt")
    env_dir = tmp_path / "envs" / "orb"
    env_dir.mkdir(parents=True)

    from rootstock.manifest import ManifestError

    def broken_load(root):
        raise ManifestError("corrupt")

    monkeypatch.setattr("rootstock.batch.load_manifest", broken_load)
    plan = PrunePlan(
        checkpoints=[
            PruneCheckpointItem("mace", "gone", "not declared", files=["cache/mace/gone.pt"])
        ],
        weight_dirs=[PruneWeightDirItem("cache/huggingface/hub/models--a--b", "released")],
        envs=[PruneEnvItem("orb", "no registered source")],
    )

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    got = statuses(report)
    assert got[("checkpoint", "mace/gone")] == "failed"
    assert got[("weights", "models--a--b")] == "skipped"
    assert got[("env", "orb")] == "ok"
    # Record-first ordering: with the record still standing, its files must survive.
    assert weight_file.exists()
    assert not env_dir.exists()
    assert len(refreshes) == 1  # the env succeeded; state changed


def test_gc_items_keep_going_past_failures(tmp_path, refreshes, monkeypatch):
    stale = tmp_path / ".build" / "mace.1"
    stale.mkdir(parents=True)
    lock = touch(tmp_path, "environments/dead.py.lock")
    monkeypatch.setattr("rootstock.batch.shutil.which", lambda _: None)  # uv "missing"
    plan = PrunePlan(
        gc=[
            GCItem("uv-cache", str(tmp_path / ".uv-cache"), "delegate"),
            GCItem("build", str(stale), "leftover"),
            GCItem("lockfile", str(lock), "orphaned"),
        ]
    )

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    got = statuses(report)
    assert got[("gc", "uv-cache:.uv-cache")] == "failed"
    assert got[("gc", "build:mace.1")] == "ok"
    assert got[("gc", "lockfile:dead.py.lock")] == "ok"
    assert not stale.exists() and not lock.exists()


def test_uv_cache_runs_uv_cache_prune_with_the_roots_cache_dir(tmp_path, refreshes, monkeypatch):
    uv_cache = tmp_path / ".uv-cache"
    uv_cache.mkdir()
    calls: list = []

    class Proc:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs["env"]["UV_CACHE_DIR"]))
        return Proc()

    monkeypatch.setattr("rootstock.batch.shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr("rootstock.batch.subprocess.run", fake_run)
    plan = PrunePlan(gc=[GCItem("uv-cache", str(uv_cache), "delegate")])

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    assert statuses(report) == {("gc", "uv-cache:.uv-cache"): "ok"}
    assert calls == [(["uv", "cache", "prune"], str(uv_cache))]


def test_execution_streams_each_action_live(tmp_path, refreshes):
    """Deletes stream as they happen (serial phases use live output, not the
    buffer-until-failure idiom sync's parallel phases need) — and each action
    is logged *before* it runs, so a hung delete shows its target."""
    save(tmp_path, {"mace": env_record({"gone": CheckpointInfo(fetched_at="x")})})
    touch(tmp_path, "cache/mace/gone.pt")
    touch(tmp_path, "cache/huggingface/hub/models--a--b/blobs/aa")
    env_dir = tmp_path / "envs" / "orb"
    env_dir.mkdir(parents=True)
    plan = PrunePlan(
        checkpoints=[
            PruneCheckpointItem("mace", "gone", "not declared", files=["cache/mace/gone.pt"])
        ],
        weight_dirs=[PruneWeightDirItem("cache/huggingface/hub/models--a--b", "released")],
        envs=[PruneEnvItem("orb", "no registered source")],
    )

    lines: list[str] = []
    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lines.append)

    assert not report.failed
    stripped = [line.strip() for line in lines]
    assert any(line.startswith("dropping 1 checkpoint record") for line in stripped)
    assert "rm cache/mace/gone.pt (4 B)" in stripped
    assert any(line.startswith("rm -r cache/huggingface/hub/models--a--b") for line in stripped)
    assert any(line.startswith("mv envs/orb -> .trash/orb.") for line in stripped)
    assert any(line.startswith("refreshing manifest") for line in stripped)
    # The live lines land *before* the item's completion mark.
    assert stripped.index("rm cache/mace/gone.pt (4 B)") < next(
        i for i, line in enumerate(stripped) if line.startswith("[checkpoint 1/1]")
    )


def test_missing_files_are_tolerated_for_idempotent_reruns(tmp_path, refreshes):
    save(tmp_path, {"mace": env_record({"gone": CheckpointInfo(fetched_at="x")})})
    plan = PrunePlan(
        checkpoints=[
            PruneCheckpointItem(
                "mace", "gone", "not declared", files=["cache/mace/already-gone.pt"]
            )
        ]
    )

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    assert statuses(report) == {("checkpoint", "mace/gone"): "ok"}


def test_nothing_succeeded_means_no_refresh(tmp_path, refreshes, monkeypatch):
    monkeypatch.setattr("rootstock.batch.shutil.which", lambda _: None)
    plan = PrunePlan(gc=[GCItem("uv-cache", str(tmp_path / ".uv-cache"), "delegate")])

    report = execute_prune(tmp_path, plan, cache_root=tmp_path, say=lambda _: None)

    assert [r.status for r in report.results] == ["failed"]
    assert refreshes == []


def test_fail_fast_skips_later_phases(tmp_path, refreshes, monkeypatch):
    monkeypatch.setattr("rootstock.batch.shutil.which", lambda _: None)
    env_dir = tmp_path / "envs" / "orb"
    env_dir.mkdir(parents=True)
    save(tmp_path, {"mace": env_record({"gone": CheckpointInfo(fetched_at="x")})})

    from rootstock.manifest import ManifestError

    def broken_load(root):
        raise ManifestError("corrupt")

    monkeypatch.setattr("rootstock.batch.load_manifest", broken_load)
    plan = PrunePlan(
        checkpoints=[PruneCheckpointItem("mace", "gone", "not declared")],
        envs=[PruneEnvItem("orb", "no registered source")],
        gc=[GCItem("build", str(tmp_path / ".build" / "x"), "leftover")],
    )

    report = execute_prune(tmp_path, plan, fail_fast=True, cache_root=tmp_path, say=lambda _: None)

    got = statuses(report)
    assert got[("checkpoint", "mace/gone")] == "failed"
    assert got[("env", "orb")] == "skipped"
    assert got[("gc", "build:x")] == "skipped"
    assert env_dir.exists()
    assert refreshes == []
