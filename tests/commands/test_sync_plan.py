"""The ``rootstock sync`` planner.

The plan is the diff between declared state (registered env sources,
optionally overlaid by a staging directory, plus their CHECKPOINTS tables)
and actual state (built envs + manifest). These tests pin the trigger table
from issue #182: what gets built, downloaded, and verified — and what is
deliberately left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import rootstock
from rootstock.batch import plan_sync
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    VerificationRecord,
    compute_source_hash,
    create_manifest,
    save_manifest,
)
from rootstock.operations import OperationError

OLD = "2026-07-01T00:00:00+00:00"
NEWER = "2026-07-02T00:00:00+00:00"


def env_source(*checkpoints: str, custom: str | None = None) -> str:
    entries = "".join(f"    {ckpt!r}: {ckpt!r},\n" for ckpt in checkpoints)
    if custom:
        entries += f"    {custom!r}: None,\n"
    body = (
        f"CHECKPOINTS = {{\n{entries}}}\n\ndef setup(checkpoint, device='cuda'):\n    return None\n"
    )
    if custom:
        body += "\ndef setup_from_path(path, device='cuda'):\n    return None\n"
    return body


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
    root: Path,
    name: str,
    *,
    built_at: str = OLD,
    checkpoints: dict[str, CheckpointInfo] | None = None,
    rootstock_version: str | None = None,
) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at=built_at,
        source_hash=compute_source_hash(root / "envs" / name / "env_source.py"),
        source="",
        python_requires=">=3.11",
        dependencies={"rootstock": rootstock_version or rootstock.__version__},
        checkpoints=checkpoints or {},
    )


def save(root: Path, environments: dict[str, EnvironmentInfo]) -> None:
    from rootstock.config import UserConfig

    manifest = create_manifest(root, ["test"], UserConfig(name="t", email="t@t.t"))
    manifest.environments = environments
    save_manifest(manifest, root)


VERIFIED = CheckpointInfo(
    fetched_at=NEWER,
    verifications={"test": VerificationRecord(verified_at=NEWER, verified_device="cuda")},
)


@pytest.fixture
def converged(tmp_path: Path) -> Path:
    """One env, registered + built from the same source, checkpoint fetched
    and verified after the build: the plan must be empty."""
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    save(tmp_path, {"mace": record(tmp_path, "mace", checkpoints={"mace-mp-0-medium": VERIFIED})})
    return tmp_path


def test_converged_root_plans_nothing(converged):
    plan = plan_sync(converged)
    assert plan.is_empty
    assert plan.notes == []


def test_fresh_root_with_staging_builds_everything(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "mace.py").write_text(env_source("mace-mp-0-medium", "mace-mp-0-small"))
    (staging / "uma.py").write_text(env_source("uma-s-1p1"))
    save(tmp_path, {})

    plan = plan_sync(tmp_path, source_dir=staging)

    assert [(b.env_name, b.reason) for b in plan.builds] == [
        ("mace", "not built"),
        ("uma", "not built"),
    ]
    assert plan.builds[0].source == str(staging / "mace.py")
    assert {(d.env_name, d.checkpoint) for d in plan.downloads} == {
        ("mace", "mace-mp-0-medium"),
        ("mace", "mace-mp-0-small"),
        ("uma", "uma-s-1p1"),
    }
    assert all(v.reason == "stale after rebuild" for v in plan.verifies)
    assert {v.checkpoint for v in plan.verifies} == {
        "mace-mp-0-medium",
        "mace-mp-0-small",
        "uma-s-1p1",
    }


def test_registered_but_unbuilt_env_builds_by_name(tmp_path):
    register(tmp_path, "mace", env_source("mace-mp-0-medium"))
    save(tmp_path, {})

    plan = plan_sync(tmp_path)

    (item,) = plan.builds
    assert (item.env_name, item.source, item.reason) == ("mace", "mace", "not built")


def test_staged_source_change_triggers_rebuild(converged):
    staging = converged / "staging"
    staging.mkdir()
    (staging / "mace.py").write_text(env_source("mace-mp-0-medium", "mace-off23-small"))

    plan = plan_sync(converged, source_dir=staging)

    (item,) = plan.builds
    assert (item.env_name, item.reason) == ("mace", "source changed")
    assert item.source == str(staging / "mace.py")
    # The new checkpoint downloads; both go stale once the env rebuilds.
    assert [(d.checkpoint, d.reason) for d in plan.downloads] == [
        ("mace-off23-small", "not fetched")
    ]
    assert {(v.checkpoint, v.reason) for v in plan.verifies} == {
        ("mace-mp-0-medium", "stale after rebuild"),
        ("mace-off23-small", "stale after rebuild"),
    }


def test_identical_staged_source_is_not_a_rebuild(converged):
    staging = converged / "staging"
    staging.mkdir()
    (staging / "mace.py").write_text(env_source("mace-mp-0-medium"))

    assert plan_sync(converged, source_dir=staging).is_empty


def test_rebuild_flag_rebuilds_and_restales(converged):
    plan = plan_sync(converged, rebuild=True)

    (item,) = plan.builds
    assert (item.env_name, item.reason) == ("mace", "--rebuild")
    assert plan.downloads == []  # weights survive rebuilds
    assert [(v.checkpoint, v.reason) for v in plan.verifies] == [
        ("mace-mp-0-medium", "stale after rebuild")
    ]


def test_fetched_but_never_verified_plans_verify_only(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    save(
        tmp_path,
        {
            "mace": record(
                tmp_path,
                "mace",
                checkpoints={"mace-mp-0-medium": CheckpointInfo(fetched_at=NEWER)},
            )
        },
    )

    plan = plan_sync(tmp_path)

    assert plan.builds == [] and plan.downloads == []
    assert [(v.checkpoint, v.reason) for v in plan.verifies] == [
        ("mace-mp-0-medium", "never verified")
    ]


def test_verified_before_last_build_is_stale(tmp_path):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    stale = CheckpointInfo(
        fetched_at=OLD,
        verifications={"test": VerificationRecord(verified_at=OLD, verified_device="cuda")},
    )
    save(
        tmp_path,
        {"mace": record(tmp_path, "mace", built_at=NEWER, checkpoints={"mace-mp-0-medium": stale})},
    )

    plan = plan_sync(tmp_path)

    assert [(v.checkpoint, v.reason) for v in plan.verifies] == [
        ("mace-mp-0-medium", "stale (verified before last build)")
    ]


def test_custom_checkpoints_are_ignored(tmp_path):
    source = env_source("uma-s-1p1", custom="uma:custom")
    register(tmp_path, "uma", source)
    save(tmp_path, {})

    plan = plan_sync(tmp_path)

    assert {d.checkpoint for d in plan.downloads} == {"uma-s-1p1"}
    assert {v.checkpoint for v in plan.verifies} == {"uma-s-1p1"}


def test_env_filter_limits_and_validates(converged):
    register(converged, "uma", env_source("uma-s-1p1"))

    plan = plan_sync(converged, envs=["uma"])
    assert [b.env_name for b in plan.builds] == ["uma"]
    assert {d.env_name for d in plan.downloads} == {"uma"}

    with pytest.raises(OperationError, match="Unknown environment"):
        plan_sync(converged, envs=["nonexistent"])


def test_checkpoint_filter_limits_and_validates(tmp_path):
    register(tmp_path, "mace", env_source("mace-mp-0-medium", "mace-mp-0-small"))
    save(tmp_path, {})

    plan = plan_sync(tmp_path, checkpoints=["mace-mp-0-small"])
    assert [d.checkpoint for d in plan.downloads] == ["mace-mp-0-small"]
    assert [v.checkpoint for v in plan.verifies] == ["mace-mp-0-small"]

    with pytest.raises(OperationError, match="Unknown checkpoint"):
        plan_sync(tmp_path, checkpoints=["not-a-real-id"])


def test_pin_drift_is_a_note_not_a_rebuild(tmp_path, monkeypatch):
    source = env_source("mace-mp-0-medium")
    register(tmp_path, "mace", source)
    build(tmp_path, "mace", source)
    save(
        tmp_path,
        {
            "mace": record(
                tmp_path,
                "mace",
                checkpoints={"mace-mp-0-medium": VERIFIED},
                rootstock_version="1.0.0",
            )
        },
    )
    monkeypatch.setattr(rootstock, "__version__", "2.0.0")

    plan = plan_sync(tmp_path)
    assert plan.builds == []
    assert any("built with rootstock 1.0.0" in note for note in plan.notes)

    # --rebuild acts on it, and the note disappears (no longer advisory).
    plan = plan_sync(tmp_path, rebuild=True)
    assert [b.env_name for b in plan.builds] == ["mace"]
    assert not any("built with rootstock" in note for note in plan.notes)


def test_phase_selection_drops_items(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "mace.py").write_text(env_source("mace-mp-0-medium"))
    save(tmp_path, {})

    plan = plan_sync(tmp_path, source_dir=staging, phases=("build", "download"))
    assert plan.builds and plan.downloads
    assert plan.verifies == []

    plan = plan_sync(tmp_path, source_dir=staging, phases=("verify",))
    assert plan.builds == [] and plan.downloads == []
    assert plan.verifies


def test_empty_staging_dir_is_an_error(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    with pytest.raises(OperationError, match="No .*environment files"):
        plan_sync(tmp_path, source_dir=staging)
