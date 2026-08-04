"""``rootstock sync`` on shared installs (#208).

The planner's cluster behavior, mirroring the smoke-test coverage in
tests/cli/test_smoke_test_clusters.py: verification is judged and planned
per cluster; envs restricted by CLUSTERS are still built and downloaded
(shared artifacts) but only verified by a machine they serve; and the
command refuses to guess which machine it is on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.batch import plan_sync
from rootstock.config import UserConfig
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    VerificationRecord,
    compute_source_hash,
    create_manifest,
    save_manifest,
)
from rootstock.operations import OperationError

BUILT = "2026-07-01T00:00:00+00:00"
VERIFIED = "2026-07-02T00:00:00+00:00"


def env_source(*checkpoints: str, clusters: list[str] | None = None, malformed=False) -> str:
    body = ""
    if malformed:
        body += "CLUSTERS = []\n"
    elif clusters is not None:
        body += f"CLUSTERS = {clusters!r}\n"
    entries = "".join(f"    {ckpt!r}: {ckpt!r},\n" for ckpt in checkpoints)
    body += (
        f"CHECKPOINTS = {{\n{entries}}}\n\ndef setup(checkpoint, device='cuda'):\n    return None\n"
    )
    return body


def register_and_build(root: Path, name: str, source: str, built: bool = True) -> None:
    env_file = root / "environments" / f"{name}.py"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(source)
    if built:
        env_dir = root / "envs" / name
        (env_dir / "bin").mkdir(parents=True, exist_ok=True)
        (env_dir / "bin" / "python").touch()
        (env_dir / "env_source.py").write_text(source)


def record(root: Path, name: str, checkpoints: dict[str, CheckpointInfo]) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at=BUILT,
        source_hash=compute_source_hash(root / "envs" / name / "env_source.py"),
        source="",
        python_requires=">=3.11",
        dependencies={},
        checkpoints=checkpoints,
    )


def save(root: Path, environments: dict[str, EnvironmentInfo]) -> None:
    manifest = create_manifest(root, ["sophia", "polaris"], UserConfig(name="t", email="t@t.t"))
    manifest.environments = environments
    save_manifest(manifest, root)


def sophia_verified(fetched_at: str = VERIFIED) -> CheckpointInfo:
    return CheckpointInfo(
        fetched_at=fetched_at,
        verifications={"sophia": VerificationRecord(verified_at=VERIFIED, verified_device="cuda")},
    )


def test_verified_on_sophia_still_plans_verify_for_polaris(tmp_path):
    source = env_source("mace-mp-0-medium")
    register_and_build(tmp_path, "mace", source)
    save(tmp_path, {"mace": record(tmp_path, "mace", {"mace-mp-0-medium": sophia_verified()})})

    polaris = plan_sync(tmp_path, cluster="polaris")
    assert [(v.env_name, v.checkpoint, v.reason) for v in polaris.verifies] == [
        ("mace", "mace-mp-0-medium", "never verified")
    ]
    assert polaris.downloads == []  # the fetch is shared; only the verify is owed

    # sophia's own record is current — nothing to do there.
    assert plan_sync(tmp_path, cluster="sophia").is_empty


def test_restricted_env_verifies_only_where_served(tmp_path):
    source = env_source("sevennet-0", clusters=["sophia"])
    register_and_build(tmp_path, "sevennet", source)
    save(
        tmp_path,
        {
            "sevennet": record(
                tmp_path, "sevennet", {"sevennet-0": CheckpointInfo(fetched_at=VERIFIED)}
            )
        },
    )

    polaris = plan_sync(tmp_path, cluster="polaris")
    assert polaris.verifies == []
    assert any("serves only sophia" in note for note in polaris.notes)

    sophia = plan_sync(tmp_path, cluster="sophia")
    assert [(v.env_name, v.checkpoint) for v in sophia.verifies] == [("sevennet", "sevennet-0")]


def test_restricted_env_still_builds_and_downloads_everywhere(tmp_path):
    # Builds and downloads produce shared artifacts (one filesystem, one
    # weights cache) — polaris's sync still does that work for a sophia-only
    # env; only the verify belongs to sophia's chain.
    source = env_source("sevennet-0", clusters=["sophia"])
    register_and_build(tmp_path, "sevennet", source, built=False)
    save(tmp_path, {})

    polaris = plan_sync(tmp_path, cluster="polaris")
    assert [b.env_name for b in polaris.builds] == ["sevennet"]
    assert [(d.env_name, d.checkpoint) for d in polaris.downloads] == [("sevennet", "sevennet-0")]
    assert polaris.verifies == []
    assert any("serves only sophia" in note for note in polaris.notes)


def test_malformed_clusters_skips_verifies_with_note(tmp_path):
    source = env_source("mace-mp-0-medium", malformed=True)
    register_and_build(tmp_path, "mace", source)
    save(
        tmp_path,
        {
            "mace": record(
                tmp_path, "mace", {"mace-mp-0-medium": CheckpointInfo(fetched_at=VERIFIED)}
            )
        },
    )

    plan = plan_sync(tmp_path, cluster="polaris")
    # Never widen a broken restriction to "serves everywhere" — drop the
    # verifies and say why.
    assert plan.verifies == []
    assert any("cannot parse CLUSTERS" in note for note in plan.notes)


def test_plan_sync_shared_install_requires_cluster(tmp_path):
    source = env_source("mace-mp-0-medium")
    register_and_build(tmp_path, "mace", source)
    save(tmp_path, {"mace": record(tmp_path, "mace", {"mace-mp-0-medium": sophia_verified()})})

    with pytest.raises(OperationError, match="sophia, polaris"):
        plan_sync(tmp_path)

    # Without the verify phase there is nothing to attribute — no cluster needed.
    plan = plan_sync(tmp_path, phases=("build", "download"))
    assert plan.is_empty


def test_cmd_sync_exits_2_without_cluster_on_shared_install(tmp_path, monkeypatch, capsys):
    from rootstock.commands.sync import cmd_sync

    save(tmp_path, {})

    planned = []
    monkeypatch.setattr("rootstock.commands.sync.plan_sync", lambda *a, **kw: planned.append(1))

    class _Args:
        pass

    args = _Args()
    args.source_dir = None
    args.root = str(tmp_path)
    args.cluster = None
    args.env = None
    args.checkpoint = None
    args.rebuild = False
    args.upgrade = False
    args.phases = "build,download,verify"
    args.jobs = 4
    args.verify_jobs = 1
    args.verify_timeout = 600.0
    args.device = "cuda"
    args.dry_run = False
    args.json = False
    args.fail_fast = False
    args.no_push = True
    args.no_perm_check = True

    rc = cmd_sync(args)

    assert rc == 2
    err = capsys.readouterr().err
    assert "sophia" in err and "polaris" in err and "--cluster" in err
    assert planned == []  # refused before planning anything
