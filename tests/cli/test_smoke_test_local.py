"""Tests for ``rootstock smoke-test`` over local (user-registered) checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock import local_checkpoints
from rootstock.commands import smoke_test as smoke_module
from rootstock.commands.smoke_test import cmd_smoke_test
from rootstock.config import UserConfig
from rootstock.local_checkpoints import (
    local_checkpoints_for_root,
    register_local_checkpoint,
)
from rootstock.manifest import (
    EnvironmentInfo,
    create_manifest,
    load_manifest,
    save_manifest,
)

_ENV_SOURCE = """\
CHECKPOINTS = {"uma-s-1p1": "uma-s-1p1"}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch) -> Path:
    """A root with one built env (uma) in the manifest, no fetched canonical
    checkpoints — so only local checkpoints are selectable."""
    root = tmp_path / "root"
    env_dir = root / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE)

    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(root, "test", cfg)
    manifest.environments["uma"] = EnvironmentInfo(
        built_at="2026-01-01T00:00:00+00:00",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
    )
    save_manifest(manifest, root)

    monkeypatch.setattr(
        "rootstock.commands.smoke_test.update_and_push_manifest",
        lambda *a, **kw: True,
    )
    return root


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "registry.json"
    monkeypatch.setattr(local_checkpoints, "DEFAULT_LOCAL_REGISTRY_FILE", path)
    return path


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "ft.pt"
    path.write_bytes(b"fine-tuned weights")
    return path


@pytest.fixture
def registered(fake_root, weights, registry) -> str:
    register_local_checkpoint(fake_root, "my-uma-ft", "uma", weights, setup_kwargs={"task": "omol"})
    return "my-uma-ft"


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


def test_local_checkpoint_verified_with_registered_kwargs(
    fake_root, weights, registry, registered, monkeypatch
):
    calls = []

    def fake_verify(**kwargs):
        calls.append(kwargs)
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)
    rc = cmd_smoke_test(_make_args(fake_root))
    assert rc == 0

    assert len(calls) == 1
    call = calls[0]
    assert call["env_name"] == "uma"
    assert call["checkpoint"] == "my-uma-ft"
    assert call["checkpoint_path"] == str(weights.resolve())
    # Registered kwargs, not {} — deliberate divergence from canonical policy.
    assert call["setup_kwargs"] == {"task": "omol"}

    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert entry.verified_at is not None
    assert entry.verified_device == "cuda"


def test_local_failure_recorded_in_registry_not_manifest(
    fake_root, weights, registry, registered, monkeypatch
):
    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda **kw: (False, "boom"))
    manifest_before = (fake_root / "manifest.json").read_text()

    rc = cmd_smoke_test(_make_args(fake_root))
    assert rc == 1

    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert entry.last_error == "smoke-test: boom"
    # Local-only run: the shared manifest is untouched (a non-maintainer
    # couldn't write it anyway).
    assert (fake_root / "manifest.json").read_text() == manifest_before


def test_hash_mismatch_fails_without_verify(fake_root, weights, registry, registered, monkeypatch):
    def unexpected_verify(**kw):
        raise AssertionError("verify must not run on a hash mismatch")

    monkeypatch.setattr(smoke_module, "verify_checkpoint", unexpected_verify)
    weights.write_bytes(b"silently swapped weights")

    rc = cmd_smoke_test(_make_args(fake_root))
    assert rc == 1
    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert "changed on disk" in entry.last_error


def test_missing_file_fails_without_verify(fake_root, weights, registry, registered, monkeypatch):
    def unexpected_verify(**kw):
        raise AssertionError("verify must not run on a missing file")

    monkeypatch.setattr(smoke_module, "verify_checkpoint", unexpected_verify)
    weights.unlink()

    rc = cmd_smoke_test(_make_args(fake_root))
    assert rc == 1
    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert "missing" in entry.last_error


def test_env_filter_applies_to_locals(
    fake_root, weights, registry, registered, monkeypatch, capsys
):
    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda **kw: (True, None))
    rc = cmd_smoke_test(_make_args(fake_root, env="mace"))
    assert rc == 0
    assert "No fetched checkpoints to test." in capsys.readouterr().out


def test_checkpoint_filter_selects_local(fake_root, weights, registry, registered, monkeypatch):
    calls = []

    def fake_verify(**kwargs):
        calls.append(kwargs["checkpoint"])
        return True, None

    monkeypatch.setattr(smoke_module, "verify_checkpoint", fake_verify)
    rc = cmd_smoke_test(_make_args(fake_root, env="uma", checkpoint="my-uma-ft"))
    assert rc == 0
    assert calls == ["my-uma-ft"]


def test_json_output_tags_local_results(
    fake_root, weights, registry, registered, monkeypatch, capsys
):
    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda **kw: (True, None))
    rc = cmd_smoke_test(_make_args(fake_root, json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] == 1
    (result,) = payload["results"]
    assert result["local"] is True
    assert result["checkpoint"] == "my-uma-ft"
    assert result["verified_current"] is True


def test_local_only_run_works_without_manifest(
    fake_root, weights, registry, registered, monkeypatch
):
    # A root with no manifest but registered locals still smoke-tests them
    # (e.g. a personal install created without init).
    (fake_root / "manifest.json").unlink()
    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda **kw: (True, None))
    rc = cmd_smoke_test(_make_args(fake_root))
    assert rc == 0
    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert entry.verified_at is not None
    # Still no manifest afterwards — nothing shared was created.
    assert not (fake_root / "manifest.json").exists()


def test_manifest_updated_when_canonical_also_selected(
    fake_root, weights, registry, registered, monkeypatch
):
    # Add a fetched canonical checkpoint; both it and the local one run, and
    # the manifest write happens for the canonical outcome.
    from rootstock.manifest import CheckpointInfo

    manifest = load_manifest(fake_root)
    manifest.environments["uma"].checkpoints["uma-s-1p1"] = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00+00:00"
    )
    save_manifest(manifest, fake_root)

    monkeypatch.setattr(smoke_module, "verify_checkpoint", lambda **kw: (True, None))
    rc = cmd_smoke_test(_make_args(fake_root))
    assert rc == 0

    fresh = load_manifest(fake_root)
    canonical = fresh.environments["uma"].checkpoints["uma-s-1p1"]
    assert canonical.verified_at is not None
    # The local id never leaks into the manifest.
    assert "my-uma-ft" not in fresh.environments["uma"].checkpoints
