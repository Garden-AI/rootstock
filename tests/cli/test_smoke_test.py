"""Tests for ``rootstock smoke-test``."""

from __future__ import annotations

import json
from pathlib import Path

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
    manifest = create_manifest(root, "test", cfg)

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
    args.json = overrides.get("json", False)
    args.root = str(root)
    args.no_push = overrides.get("no_push", True)
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
    assert medium.verified_at is None
    assert "smoke-test:" in medium.last_error
    assert small.verified_at is not None
    assert small.last_error is None


def test_smoke_test_returns_zero_when_all_pass(populated_root, monkeypatch):
    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda *a, **kw: (True, None))
    assert cmd_smoke_test(_make_args(populated_root)) == 0


def test_smoke_test_filters_by_env(populated_root, monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        smoke_module, "verify_checkpoint",
        lambda root, env_name, checkpoint, device, setup_kwargs, **_: (
            seen.append((env_name, checkpoint)) or (True, None)
        ),
    )

    cmd_smoke_test(_make_args(populated_root, env="uma"))
    assert seen == [("uma", "uma-s-1p1")]


def test_smoke_test_filters_by_env_and_checkpoint(populated_root, monkeypatch):
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        smoke_module, "verify_checkpoint",
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
    assert parsed["passed"] >= 1
    assert parsed["failed"] == 0
    assert isinstance(parsed["results"], list)
    assert all("verified_current" in r for r in parsed["results"])


def test_smoke_test_no_manifest_returns_1(tmp_path, capsys):
    rc = cmd_smoke_test(_make_args(tmp_path))
    assert rc == 1


def test_smoke_test_empty_selection_returns_0(tmp_path, monkeypatch):
    """Manifest exists but has no fetched checkpoints — clean exit, no failures."""
    cfg = UserConfig(name="t", email="t@t.t")
    save_manifest(create_manifest(tmp_path, "test", cfg), tmp_path)
    monkeypatch.setattr(
        "rootstock.commands.smoke_test.update_and_push_manifest", lambda *a, **kw: True
    )
    rc = cmd_smoke_test(_make_args(tmp_path))
    assert rc == 0
