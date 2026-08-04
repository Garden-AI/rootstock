"""Tests for ``rootstock smoke-test``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rootstock.commands import smoke_test as smoke_module
from rootstock.commands.smoke_test import cmd_smoke_test
from rootstock.config import UserConfig
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    create_manifest,
    load_manifest,
    save_manifest,
)


@pytest.fixture
def populated_root(tmp_path: Path, monkeypatch) -> Path:
    """A root with two envs: mace (small + medium fetched, large NOT fetched)
    and uma (one fetched checkpoint). Built dirs exist so verify can be called."""
    root = tmp_path
    for env in ("mace", "uma"):
        (root / "envs" / env / "bin").mkdir(parents=True)
        (root / "envs" / env / "bin" / "python").touch()

    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(root, ["test"], cfg)

    manifest.environments["mace"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints={
            "mace-mp-0-small": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
            "mace-mp-0-medium": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
            "mace-mp-0-large": CheckpointInfo(),  # not fetched yet — should be skipped
        },
    )
    manifest.environments["uma"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:def",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints={
            "uma-s-1p1": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
        },
    )
    save_manifest(manifest, root)

    monkeypatch.setattr(
        "rootstock.commands.smoke_test.update_and_push_manifest",
        lambda *a, **kw: True,
    )
    return root


def _make_args(root: Path, **overrides):
    class _Args:
        pass

    args = _Args()
    args.env = overrides.get("env")
    args.checkpoint = overrides.get("checkpoint")
    args.device = overrides.get("device", "cuda")
    args.verify_timeout = overrides.get("verify_timeout", 600.0)
    args.json = overrides.get("json", False)
    args.root = str(root)
    args.no_push = overrides.get("no_push", True)
    args.cluster = overrides.get("cluster")
    return args


def test_smoke_test_skips_unfetched_checkpoints(populated_root, monkeypatch, capsys):
    seen: list[tuple[str, str]] = []

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **_):
        seen.append((env_name, checkpoint))
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)

    rc = cmd_smoke_test(_make_args(populated_root))
    assert rc == 0

    # mace/mace-mp-0-large was not fetched, so it should not have been tested.
    assert sorted(seen) == [
        ("mace", "mace-mp-0-medium"),
        ("mace", "mace-mp-0-small"),
        ("uma", "uma-s-1p1"),
    ]


def test_smoke_test_always_uses_empty_kwargs(populated_root, monkeypatch):
    captured: list[dict] = []

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **_):
        captured.append(setup_kwargs)
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)

    cmd_smoke_test(_make_args(populated_root))
    assert captured  # at least one verify happened
    assert all(k == {} for k in captured)


def test_smoke_test_forwards_verify_timeout(populated_root, monkeypatch):
    captured: list[float] = []

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, timeout, **_):
        captured.append(timeout)
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)

    cmd_smoke_test(_make_args(populated_root, verify_timeout=1800.0))
    assert captured  # at least one verify happened
    assert all(t == 1800.0 for t in captured)


def test_smoke_test_marks_pass_and_fail(populated_root, monkeypatch):
    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **_):
        if (env_name, checkpoint) == ("mace", "mace-mp-0-medium"):
            return False, "RuntimeError: bad"
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)
    rc = cmd_smoke_test(_make_args(populated_root))
    assert rc == 1  # at least one failure -> exit 1

    m = load_manifest(populated_root)
    medium = m.environments["mace"].checkpoints["mace-mp-0-medium"]
    small = m.environments["mace"].checkpoints["mace-mp-0-small"]
    assert medium.verification("test").verified_at is None
    assert "smoke-test:" in medium.verification("test").last_error
    assert small.verification("test").verified_at is not None
    assert small.verification("test").last_error is None


def test_smoke_test_returns_zero_when_all_pass(populated_root, monkeypatch):
    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda *a, **kw: (True, None))
    assert cmd_smoke_test(_make_args(populated_root)) == 0


def test_smoke_test_rerecords_weight_files(populated_root, monkeypatch):
    """Every pass re-captures the weight files the load touched, so records
    self-heal nightly like verified_at (#177) — and one smoke-test run
    backfills an install that predates weight tracking."""
    files = [{"path": "cache/fake/model.bin", "size": 9_000_000}]

    def fake_verify(
        root, env_name, checkpoint, device, setup_kwargs, *, weights_capture_path=None, **_
    ):
        Path(weights_capture_path).write_text(json.dumps({"files": files}))
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)
    assert cmd_smoke_test(_make_args(populated_root)) == 0

    small = load_manifest(populated_root).environments["mace"].checkpoints["mace-mp-0-small"]
    assert small.weight_files == files
    assert small.weights_recorded_at is not None


def test_smoke_test_filters_by_env(populated_root, monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        smoke_module,
        "verify_checkpoint",
        lambda root, env_name, checkpoint, device, setup_kwargs, **_: (
            seen.append((env_name, checkpoint)) or (True, None)
        ),
    )

    cmd_smoke_test(_make_args(populated_root, env="uma"))
    assert seen == [("uma", "uma-s-1p1")]


def test_smoke_test_filters_by_env_and_checkpoint(populated_root, monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        smoke_module,
        "verify_checkpoint",
        lambda root, env_name, checkpoint, device, setup_kwargs, **_: (
            seen.append((env_name, checkpoint)) or (True, None)
        ),
    )

    cmd_smoke_test(_make_args(populated_root, env="mace", checkpoint="mace-mp-0-small"))
    assert seen == [("mace", "mace-mp-0-small")]


def test_smoke_test_checkpoint_without_env_errors(populated_root, capsys):
    rc = cmd_smoke_test(_make_args(populated_root, checkpoint="mace-mp-0-small"))
    assert rc == 2


def test_smoke_test_emits_valid_json(populated_root, monkeypatch, capsys):
    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda *a, **kw: (True, None))

    rc = cmd_smoke_test(_make_args(populated_root, json=True))
    assert rc == 0

    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)
    assert parsed["cluster"] == "test"
    assert parsed["passed"] >= 1
    assert parsed["failed"] == 0
    assert isinstance(parsed["results"], list)
    assert all("verified_current" in r for r in parsed["results"])
    assert all(r["cluster"] == "test" for r in parsed["results"])


def test_smoke_test_no_manifest_returns_1(tmp_path, capsys):
    rc = cmd_smoke_test(_make_args(tmp_path))
    assert rc == 1


def test_smoke_test_empty_selection_returns_0(tmp_path, monkeypatch):
    """Manifest exists but has no fetched checkpoints — clean exit, no failures."""
    cfg = UserConfig(name="t", email="t@t.t")
    save_manifest(create_manifest(tmp_path, ["test"], cfg), tmp_path)
    monkeypatch.setattr(
        "rootstock.commands.smoke_test.update_and_push_manifest", lambda *a, **kw: True
    )
    rc = cmd_smoke_test(_make_args(tmp_path))
    assert rc == 0


# ---------------------------------------------------------------------------
# Custom-weights legs (#200): each '<family>:custom' entry re-loads a
# same-family checkpoint's cached weights via weights= and must agree.
# ---------------------------------------------------------------------------

UMA_SOURCE = """\
CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    raise NotImplementedError


def setup_from_path(path: str, device: str = "cuda"):
    raise NotImplementedError
"""

MACE_SOURCE = """\
CHECKPOINTS = {
    "mace-mp-0-small": "small",
    "mace-off23-small": "off:small",
    "mace:custom": None,
    "mace-off:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    raise NotImplementedError


def setup_from_path(path: str, device: str = "cuda"):
    raise NotImplementedError
"""

# Weights file each canonical checkpoint's load "touches" (cache-root-relative).
CAPTURES = {
    "uma-s-1p1": "cache/uma/uma-s-1p1.pt",
    "mace-mp-0-small": "cache/mace/mp-small.model",
    "mace-off23-small": "cache/mace/off-small.model",
}


@pytest.fixture
def custom_root(tmp_path: Path, monkeypatch) -> Path:
    """A root whose built env sources declare ':custom' entries: uma (one
    family) and mace (two families), every canonical checkpoint fetched and
    its weights file present on disk."""
    root = tmp_path
    for env, source in (("uma", UMA_SOURCE), ("mace", MACE_SOURCE)):
        (root / "envs" / env / "bin").mkdir(parents=True)
        (root / "envs" / env / "bin" / "python").touch()
        (root / "envs" / env / "env_source.py").write_text(source)
    for rel in CAPTURES.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")

    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(root, ["test"], cfg)
    manifest.environments["uma"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints={"uma-s-1p1": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z")},
    )
    manifest.environments["mace"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:def",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints={
            "mace-mp-0-small": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
            "mace-off23-small": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
        },
    )
    save_manifest(manifest, root)

    monkeypatch.setattr(
        "rootstock.commands.smoke_test.update_and_push_manifest",
        lambda *a, **kw: True,
    )
    return root


def _fake_verify_factory(calls: list[dict], custom_energy_offset: float = 0.0, fail=()):
    """A verify_checkpoint stand-in: canonical calls write a one-file weight
    capture (per CAPTURES); every successful call fills identical results,
    except custom legs get ``custom_energy_offset`` added."""

    def fake_verify(
        root,
        env_name,
        checkpoint,
        device,
        setup_kwargs,
        cache_root=None,
        checkpoint_path=None,
        weights_capture_path=None,
        results=None,
        timeout=None,
    ):
        calls.append(
            {"env": env_name, "checkpoint": checkpoint, "checkpoint_path": checkpoint_path}
        )
        if checkpoint in fail:
            return False, "RuntimeError: boom"
        if weights_capture_path is not None and checkpoint in CAPTURES:
            Path(weights_capture_path).write_text(
                json.dumps({"files": [{"path": CAPTURES[checkpoint], "size": 9_000_000}]})
            )
        if results is not None:
            results["energy"] = -10.0 + (custom_energy_offset if checkpoint_path else 0.0)
            results["forces"] = np.full((3, 3), 0.1)
        return True, None

    return fake_verify


def test_custom_leg_verifies_and_records(custom_root, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(smoke_module, "verify_checkpoint", _fake_verify_factory(calls))

    assert cmd_smoke_test(_make_args(custom_root)) == 0

    m = load_manifest(custom_root)
    custom = m.environments["uma"].checkpoints["uma:custom"]
    record = custom.verification("test")
    assert record.verified_at is not None
    assert record.verified_device == "cuda"
    assert record.last_error is None
    assert custom.last_error is None
    assert custom.fetched_at is None  # never fetched — the user supplies weights

    (leg,) = [c for c in calls if c["checkpoint"] == "uma:custom"]
    assert leg["checkpoint_path"] == str(custom_root / "cache/uma/uma-s-1p1.pt")


def test_custom_leg_pairs_each_family_with_its_base(custom_root, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(smoke_module, "verify_checkpoint", _fake_verify_factory(calls))

    assert cmd_smoke_test(_make_args(custom_root, env="mace")) == 0

    by_ckpt = {c["checkpoint"]: c for c in calls if c["checkpoint_path"]}
    assert by_ckpt["mace:custom"]["checkpoint_path"] == str(
        custom_root / "cache/mace/mp-small.model"
    )
    assert by_ckpt["mace-off:custom"]["checkpoint_path"] == str(
        custom_root / "cache/mace/off-small.model"
    )

    m = load_manifest(custom_root)
    ckpts = m.environments["mace"].checkpoints
    assert ckpts["mace:custom"].verification("test").verified_at is not None
    assert ckpts["mace-off:custom"].verification("test").verified_at is not None


def test_custom_leg_divergence_fails_and_records_error(custom_root, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        smoke_module,
        "verify_checkpoint",
        _fake_verify_factory(calls, custom_energy_offset=1.0),
    )

    assert cmd_smoke_test(_make_args(custom_root, env="uma")) == 1

    custom = load_manifest(custom_root).environments["uma"].checkpoints["uma:custom"]
    assert custom.verification("test").verified_at is None
    assert "diverges" in custom.verification("test").last_error
    # The canonical baseline itself passed.
    base = load_manifest(custom_root).environments["uma"].checkpoints["uma-s-1p1"]
    assert base.verification("test").verified_at is not None


def test_custom_leg_skipped_when_baseline_fails(custom_root, monkeypatch, capsys):
    calls: list[dict] = []
    monkeypatch.setattr(
        smoke_module,
        "verify_checkpoint",
        _fake_verify_factory(calls, fail={"uma-s-1p1"}),
    )

    rc = cmd_smoke_test(_make_args(custom_root, env="uma", json=True))
    assert rc == 1  # the canonical failure

    parsed = json.loads(capsys.readouterr().out.strip())
    (skip,) = parsed["skipped"]
    assert skip["checkpoint"] == "uma:custom"
    assert "baseline" in skip["reason"]
    # No weights= call was attempted, and no manifest entry was written.
    assert all(c["checkpoint_path"] is None for c in calls)
    assert "uma:custom" not in load_manifest(custom_root).environments["uma"].checkpoints


def test_custom_leg_skipped_without_fetched_family_base(custom_root, monkeypatch, capsys):
    m = load_manifest(custom_root)
    m.environments["uma"].checkpoints["uma-s-1p1"].fetched_at = None
    save_manifest(m, custom_root)

    monkeypatch.setattr(smoke_module, "verify_checkpoint", _fake_verify_factory([]))
    rc = cmd_smoke_test(_make_args(custom_root, env="uma", json=True))
    assert rc == 0

    parsed = json.loads(capsys.readouterr().out.strip())
    assert parsed["results"] == []
    (skip,) = parsed["skipped"]
    assert skip["checkpoint"] == "uma:custom"
    assert "no fetched" in skip["reason"]


def test_custom_leg_skipped_without_dominant_weights_file(custom_root, monkeypatch, capsys):
    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **kw):
        if kw.get("weights_capture_path"):
            # Two comparable shards — no single file to hand to weights=.
            Path(kw["weights_capture_path"]).write_text(
                json.dumps(
                    {
                        "files": [
                            {"path": "cache/uma/shard1.pt", "size": 9_000_000},
                            {"path": "cache/uma/shard2.pt", "size": 8_000_000},
                        ]
                    }
                )
            )
        if kw.get("results") is not None:
            kw["results"]["energy"] = -10.0
            kw["results"]["forces"] = np.full((3, 3), 0.1)
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)
    rc = cmd_smoke_test(_make_args(custom_root, env="uma", json=True))
    assert rc == 0

    parsed = json.loads(capsys.readouterr().out.strip())
    (skip,) = parsed["skipped"]
    assert "dominant" in skip["reason"]
    assert "uma:custom" not in load_manifest(custom_root).environments["uma"].checkpoints


def test_custom_checkpoint_filter_runs_only_leg_and_baseline(custom_root, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(smoke_module, "verify_checkpoint", _fake_verify_factory(calls))

    rc = cmd_smoke_test(_make_args(custom_root, env="uma", checkpoint="uma:custom"))
    assert rc == 0
    assert [c["checkpoint"] for c in calls] == ["uma-s-1p1", "uma:custom"]


def test_select_never_picks_custom_manifest_entries(custom_root):
    """Even a hand-edited ':custom' record with fetched_at set must not enter
    the canonical loop — there are no shipped weights for setup() to load."""
    m = load_manifest(custom_root)
    m.environments["uma"].checkpoints["uma:custom"] = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z"
    )
    selected = smoke_module._select(custom_root, m, None, None, "test")
    assert all(name != "uma:custom" for _, name, _, _ in selected)


def test_family_of_longest_match_wins():
    families = ["mace", "mace-off"]
    assert smoke_module._family_of("mace-off23-small", families) == "mace-off"
    assert smoke_module._family_of("mace-mp-0-small", families) == "mace"
    assert smoke_module._family_of("orb-v2", ["orb-v2"]) == "orb-v2"
    assert smoke_module._family_of("esen-sm-direct-all-omol", ["uma"]) is None
    # Prefixes only count on a '-' boundary.
    assert smoke_module._family_of("macex-1", ["mace"]) is None


def test_dominant_weights_file():
    assert smoke_module._dominant_weights_file(None) is None
    assert smoke_module._dominant_weights_file([]) is None
    big = {"path": "cache/a/model.pt", "size": 9_000_000}
    small = {"path": "cache/a/config.json", "size": 1_000}
    assert smoke_module._dominant_weights_file([big, small]) == "cache/a/model.pt"
    # Two comparable shards: ambiguous, setup_from_path takes one path.
    shard = {"path": "cache/a/shard2.pt", "size": 8_000_000}
    assert smoke_module._dominant_weights_file([big, shard]) is None
    # A lone tiny file is a broken capture, not weights.
    assert smoke_module._dominant_weights_file([small]) is None
