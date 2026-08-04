"""``rootstock smoke-test`` on shared installs (#208).

The command must refuse to guess which machine it is on, record outcomes
under the named cluster without touching the sibling's records, and skip
envs restricted to other clusters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.commands.smoke_test import cmd_smoke_test
from rootstock.config import UserConfig
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    VerificationRecord,
    create_manifest,
    load_manifest,
    save_manifest,
)

SOPHIA_STAMP = "2026-07-01T00:00:00+00:00"


@pytest.fixture
def shared_root(tmp_path: Path, monkeypatch) -> Path:
    """A sophia/polaris install: one universal env with a checkpoint already
    verified on sophia, and one sophia-only env."""
    root = tmp_path
    for env in ("mace", "sevennet"):
        (root / "envs" / env / "bin").mkdir(parents=True)
        (root / "envs" / env / "bin" / "python").touch()

    manifest = create_manifest(root, ["sophia", "polaris"], UserConfig(name="t", email="t@t.t"))
    manifest.environments["mace"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints={
            "mace-mp-0-medium": CheckpointInfo(
                fetched_at="2026-01-02T00:00:00Z",
                verifications={
                    "sophia": VerificationRecord(verified_at=SOPHIA_STAMP, verified_device="cuda")
                },
            ),
        },
    )
    manifest.environments["sevennet"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:def",
        source="",
        python_requires=">=3.10",
        dependencies={},
        clusters=["sophia"],
        checkpoints={
            "sevennet-0": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
        },
    )
    save_manifest(manifest, root)

    monkeypatch.setattr(
        "rootstock.commands.smoke_test.update_and_push_manifest",
        lambda *a, **kw: True,
    )
    return root


def _args(root: Path, **overrides):
    class _Args:
        pass

    args = _Args()
    args.env = overrides.get("env")
    args.checkpoint = overrides.get("checkpoint")
    args.device = overrides.get("device", "cuda")
    args.verify_timeout = overrides.get("verify_timeout", 600.0)
    args.json = overrides.get("json", False)
    args.root = str(root)
    args.no_push = True
    args.cluster = overrides.get("cluster")
    return args


def _stub_verify(monkeypatch, ok=True, err=None):
    calls: list[tuple[str, str]] = []

    def fake_verify(*, root, env_name, checkpoint, device, setup_kwargs, cache_root, **kw):
        calls.append((env_name, checkpoint))
        return ok, err

    monkeypatch.setattr("rootstock.commands.smoke_test.verify_checkpoint", fake_verify)
    return calls


def test_shared_install_requires_cluster(shared_root, monkeypatch, capsys):
    _stub_verify(monkeypatch)
    rc = cmd_smoke_test(_args(shared_root))
    assert rc == 2
    err = capsys.readouterr().err
    assert "sophia" in err and "polaris" in err and "--cluster" in err


def test_unknown_cluster_is_rejected(shared_root, monkeypatch, capsys):
    _stub_verify(monkeypatch)
    rc = cmd_smoke_test(_args(shared_root, cluster="frontier"))
    assert rc == 2
    assert "not one this install serves" in capsys.readouterr().err


def test_polaris_run_records_under_polaris_only(shared_root, monkeypatch):
    _stub_verify(monkeypatch)
    rc = cmd_smoke_test(_args(shared_root, cluster="polaris"))
    assert rc == 0

    ckpt = load_manifest(shared_root).environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.verification("polaris").verified_at is not None
    # sophia's earlier result must survive the polaris run untouched.
    assert ckpt.verification("sophia").verified_at == SOPHIA_STAMP


def test_polaris_failure_does_not_unverify_sophia(shared_root, monkeypatch):
    _stub_verify(monkeypatch, ok=False, err="CUDA OOM")
    rc = cmd_smoke_test(_args(shared_root, cluster="polaris"))
    assert rc == 1

    ckpt = load_manifest(shared_root).environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.verification("polaris").verified_at is None
    assert ckpt.verification("polaris").last_error == "smoke-test: CUDA OOM"
    assert ckpt.verification("sophia").verified_at == SOPHIA_STAMP


def test_cluster_restricted_env_is_skipped_elsewhere(shared_root, monkeypatch):
    calls = _stub_verify(monkeypatch)

    cmd_smoke_test(_args(shared_root, cluster="polaris"))
    assert ("sevennet", "sevennet-0") not in calls

    calls.clear()
    cmd_smoke_test(_args(shared_root, cluster="sophia"))
    assert ("sevennet", "sevennet-0") in calls


def test_single_cluster_install_needs_no_flag(tmp_path, monkeypatch):
    root = tmp_path
    (root / "envs" / "mace" / "bin").mkdir(parents=True)
    (root / "envs" / "mace" / "bin" / "python").touch()
    manifest = create_manifest(root, ["della"], UserConfig(name="t", email="t@t.t"))
    manifest.environments["mace"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints={"mace-mp-0-medium": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z")},
    )
    save_manifest(manifest, root)
    monkeypatch.setattr(
        "rootstock.commands.smoke_test.update_and_push_manifest", lambda *a, **kw: True
    )
    _stub_verify(monkeypatch)

    rc = cmd_smoke_test(_args(root))
    assert rc == 0
    ckpt = load_manifest(root).environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.verification("della").verified_at is not None
