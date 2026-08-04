"""``rootstock smoke-test`` on shared installs (#208).

The command must refuse to guess which machine it is on, record outcomes
under the named cluster without touching the sibling's records, and select
checkpoint-first: each canonical id is tested in the env it resolves to on
the current cluster, so a cluster-specific variant shadows the universal
env per id.
"""

from __future__ import annotations

import json
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
FETCH_STAMP = "2026-01-02T00:00:00Z"

MACE_SOURCE = """CHECKPOINTS = {"mace-mp-0-small": "small", "mace-mp-0-medium": "medium"}

def setup(checkpoint, device="cuda"):
    return None
"""

MACE_POLARIS_SOURCE = """CLUSTERS = ["polaris"]
CHECKPOINTS = {"mace-mp-0-medium": "medium"}

def setup(checkpoint, device="cuda"):
    return None
"""

SEVENNET_SOURCE = """CLUSTERS = ["sophia"]
CHECKPOINTS = {"sevennet-0": "0"}

def setup(checkpoint, device="cuda"):
    return None
"""


def _install(root: Path, name: str, source: str) -> None:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(source)


def _env_info(clusters: list[str] | None = None, checkpoints=None) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
        clusters=clusters,
        checkpoints=checkpoints or {},
    )


@pytest.fixture
def shared_root(tmp_path: Path, monkeypatch) -> Path:
    """A sophia/polaris install: universal mace (small + medium fetched,
    medium already verified on sophia), a polaris-only variant overriding
    medium (built, but with no manifest records yet), and a sophia-only
    sevennet."""
    root = tmp_path
    _install(root, "mace", MACE_SOURCE)
    _install(root, "mace-polaris", MACE_POLARIS_SOURCE)
    _install(root, "sevennet", SEVENNET_SOURCE)

    manifest = create_manifest(root, ["sophia", "polaris"], UserConfig(name="t", email="t@t.t"))
    manifest.environments["mace"] = _env_info(
        checkpoints={
            "mace-mp-0-small": CheckpointInfo(fetched_at=FETCH_STAMP),
            "mace-mp-0-medium": CheckpointInfo(
                fetched_at=FETCH_STAMP,
                verifications={
                    "sophia": VerificationRecord(verified_at=SOPHIA_STAMP, verified_device="cuda")
                },
            ),
        },
    )
    manifest.environments["mace-polaris"] = _env_info(clusters=["polaris"])
    manifest.environments["sevennet"] = _env_info(
        clusters=["sophia"],
        checkpoints={"sevennet-0": CheckpointInfo(fetched_at=FETCH_STAMP)},
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

    manifest = load_manifest(shared_root)
    # The overridden id lands on the variant's record (checkpoint-first);
    # the universal env's record gains no polaris entry.
    variant = manifest.environments["mace-polaris"].checkpoints["mace-mp-0-medium"]
    assert variant.verification("polaris").verified_at is not None
    medium = manifest.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert medium.verification("polaris").verified_at is None
    # sophia's earlier result must survive the polaris run untouched.
    assert medium.verification("sophia").verified_at == SOPHIA_STAMP


def test_polaris_failure_does_not_unverify_sophia(shared_root, monkeypatch):
    _stub_verify(monkeypatch, ok=False, err="CUDA OOM")
    rc = cmd_smoke_test(_args(shared_root, cluster="polaris"))
    assert rc == 1

    manifest = load_manifest(shared_root)
    variant = manifest.environments["mace-polaris"].checkpoints["mace-mp-0-medium"]
    assert variant.verification("polaris").verified_at is None
    assert variant.verification("polaris").last_error == "smoke-test: CUDA OOM"
    medium = manifest.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert medium.verification("sophia").verified_at == SOPHIA_STAMP


def test_variant_shadows_universal_per_id(shared_root, monkeypatch):
    """polaris tests the overridden id via the variant only; the id the
    variant doesn't declare still runs via the universal env. sophia is
    untouched by the variant."""
    calls = _stub_verify(monkeypatch)
    cmd_smoke_test(_args(shared_root, cluster="polaris"))
    assert sorted(calls) == [
        ("mace", "mace-mp-0-small"),
        ("mace-polaris", "mace-mp-0-medium"),
    ]

    calls.clear()
    cmd_smoke_test(_args(shared_root, cluster="sophia"))
    assert sorted(calls) == [
        ("mace", "mace-mp-0-medium"),
        ("mace", "mace-mp-0-small"),
        ("sevennet", "sevennet-0"),
    ]


def test_variant_inherits_fetched_at_from_universal_record(shared_root, monkeypatch):
    """The variant had no record of the id at all — the shared cache means
    'fetched anywhere = fetched', and the new record inherits the donor's
    fetch stamp instead of reading 'verified but never fetched'."""
    _stub_verify(monkeypatch)
    assert cmd_smoke_test(_args(shared_root, cluster="polaris")) == 0

    variant = load_manifest(shared_root).environments["mace-polaris"]
    record = variant.checkpoints["mace-mp-0-medium"]
    assert record.fetched_at == FETCH_STAMP
    assert record.verification("polaris").verified_at is not None


def test_env_filter_means_resolves_to_that_env(shared_root, monkeypatch):
    calls = _stub_verify(monkeypatch)
    cmd_smoke_test(_args(shared_root, cluster="polaris", env="mace"))
    # medium resolves to the variant on polaris, so --env mace keeps only small.
    assert calls == [("mace", "mace-mp-0-small")]

    calls.clear()
    cmd_smoke_test(_args(shared_root, cluster="polaris", env="mace-polaris"))
    assert calls == [("mace-polaris", "mace-mp-0-medium")]


def test_ambiguous_id_is_skipped_not_fatal(shared_root, monkeypatch, capsys):
    """Two same-specificity envs declaring one id is an authoring error; the
    nightly run must report it and keep testing everything else."""
    dup = 'CHECKPOINTS = {"dup-0": "0"}\n\ndef setup(checkpoint, device="cuda"):\n    return None\n'
    _install(shared_root, "dup-a", dup)
    _install(shared_root, "dup-b", dup)
    manifest = load_manifest(shared_root)
    manifest.environments["dup-a"] = _env_info(
        checkpoints={"dup-0": CheckpointInfo(fetched_at=FETCH_STAMP)}
    )
    manifest.environments["dup-b"] = _env_info()
    save_manifest(manifest, shared_root)

    _stub_verify(monkeypatch)
    rc = cmd_smoke_test(_args(shared_root, cluster="polaris", json=True))
    assert rc == 0  # skips never affect the exit code

    parsed = json.loads(capsys.readouterr().out.strip())
    assert {r["checkpoint"] for r in parsed["results"]} == {
        "mace-mp-0-small",
        "mace-mp-0-medium",
    }
    (skip,) = parsed["skipped"]
    assert skip["checkpoint"] == "dup-0"
    assert "several envs" in skip["reason"]


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
    manifest.environments["mace"] = _env_info(
        checkpoints={"mace-mp-0-medium": CheckpointInfo(fetched_at=FETCH_STAMP)}
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
