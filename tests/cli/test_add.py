"""Tests for ``rootstock add``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock.commands import add as add_module
from rootstock.commands.add import cmd_add, parse_kwarg
from rootstock.manifest import Manifest, load_manifest


# ---------- parse_kwarg ----------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("task=omat", ("task", "omat")),
        ("charge=-1", ("charge", -1)),
        ("scale=1.5", ("scale", 1.5)),
        ("enabled=true", ("enabled", True)),
        ("disabled=false", ("disabled", False)),
        ("nothing=null", ("nothing", None)),
        ('label="hello world"', ("label", "hello world")),
        # plain words that aren't valid JSON fall back to strings
        ("name=foo bar", ("name", "foo bar")),
    ],
)
def test_parse_kwarg(spec, expected):
    assert parse_kwarg(spec) == expected


def test_parse_kwarg_rejects_no_equals():
    with pytest.raises(ValueError):
        parse_kwarg("nothere")


def test_parse_kwarg_rejects_empty_key():
    with pytest.raises(ValueError):
        parse_kwarg("=value")


# ---------- cmd_add (idempotency, error reporting) -------------------------


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch) -> Path:
    """Build a minimal rootstock root with a fake env so cmd_add can run."""
    root = tmp_path
    env_python = root / "envs" / "mace_env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.touch()

    # Initialize a manifest so we don't trigger config-loading paths.
    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest

    cfg = UserConfig(name="t", email="t@t.t")
    save_manifest(create_manifest(root, "test", cfg), root)

    # Stub update_and_push_manifest — not under test here.
    monkeypatch.setattr(
        "rootstock.commands.add.update_and_push_manifest",
        lambda *a, **kw: True,
    )
    return root


def _make_args(root: Path, **overrides):
    class _Args:
        pass

    args = _Args()
    args.env = overrides.get("env", "mace")
    args.checkpoint = overrides.get("checkpoint", "medium")
    args.kwarg = overrides.get("kwarg")
    args.device = overrides.get("device", "cuda")
    args.no_verify = overrides.get("no_verify", False)
    args.root = str(root)
    args.no_push = overrides.get("no_push", True)
    return args


def test_add_no_verify_sets_fetched_at(fake_root, monkeypatch):
    monkeypatch.setattr(add_module, "_run_download", lambda *a, **kw: (True, None))

    rc = cmd_add(_make_args(fake_root, no_verify=True))
    assert rc == 0

    m = load_manifest(fake_root)
    ckpt = m.environments["mace_env"].checkpoints["medium"]
    assert ckpt.fetched_at is not None
    assert ckpt.verified_at is None
    assert ckpt.last_error is None


def test_add_then_add_is_idempotent(fake_root, monkeypatch):
    download_calls = []
    verify_calls = []

    def fake_download(*a, **kw):
        download_calls.append((a, kw))
        return True, None

    def fake_verify(*a, **kw):
        verify_calls.append((a, kw))
        return True, None

    monkeypatch.setattr(add_module, "_run_download", fake_download)
    monkeypatch.setattr(add_module, "verify_checkpoint", fake_verify)

    # First call: --no-verify, only fetches.
    assert cmd_add(_make_args(fake_root, no_verify=True)) == 0
    # Second call: full add; should NOT re-download (idempotent), but should verify.
    assert cmd_add(_make_args(fake_root, no_verify=False)) == 0

    assert len(download_calls) == 1, "download should only happen once"
    assert len(verify_calls) == 1

    m = load_manifest(fake_root)
    ckpt = m.environments["mace_env"].checkpoints["medium"]
    assert ckpt.fetched_at is not None
    assert ckpt.verified_at is not None
    assert ckpt.verified_device == "cuda"
    assert ckpt.last_error is None


def test_add_records_download_failure_and_returns_1(fake_root, monkeypatch):
    monkeypatch.setattr(
        add_module, "_run_download",
        lambda *a, **kw: (False, "ConnectionError: hub unreachable"),
    )
    rc = cmd_add(_make_args(fake_root, no_verify=True))
    assert rc == 1

    m = load_manifest(fake_root)
    ckpt = m.environments["mace_env"].checkpoints["medium"]
    assert ckpt.fetched_at is None
    assert "ConnectionError" in ckpt.last_error
    assert ckpt.last_error.startswith("download:")


def test_add_records_verify_failure_and_returns_1(fake_root, monkeypatch):
    monkeypatch.setattr(add_module, "_run_download", lambda *a, **kw: (True, None))
    monkeypatch.setattr(
        add_module, "verify_checkpoint",
        lambda *a, **kw: (False, "RuntimeError: CUDA OOM"),
    )

    rc = cmd_add(_make_args(fake_root, no_verify=False))
    assert rc == 1

    m = load_manifest(fake_root)
    ckpt = m.environments["mace_env"].checkpoints["medium"]
    assert ckpt.fetched_at is not None  # download succeeded, was preserved
    assert ckpt.verified_at is None
    assert ckpt.verified_device is None
    assert "CUDA OOM" in ckpt.last_error
    assert ckpt.last_error.startswith("verify:")


def test_add_clears_last_error_on_success(fake_root, monkeypatch):
    """After a verify failure, a successful re-add should clear last_error."""
    # First attempt: verify fails.
    monkeypatch.setattr(add_module, "_run_download", lambda *a, **kw: (True, None))
    monkeypatch.setattr(
        add_module, "verify_checkpoint",
        lambda *a, **kw: (False, "ValueError: bad input"),
    )
    cmd_add(_make_args(fake_root, no_verify=False))

    m = load_manifest(fake_root)
    assert m.environments["mace_env"].checkpoints["medium"].last_error is not None

    # Second attempt: verify succeeds. last_error should clear.
    monkeypatch.setattr(add_module, "verify_checkpoint", lambda *a, **kw: (True, None))
    rc = cmd_add(_make_args(fake_root, no_verify=False))
    assert rc == 0

    m = load_manifest(fake_root)
    assert m.environments["mace_env"].checkpoints["medium"].last_error is None


def test_add_forwards_kwargs_to_download_and_verify(fake_root, monkeypatch):
    captured = {}

    def fake_download(root, env_name, checkpoint, setup_kwargs):
        captured["download_kwargs"] = setup_kwargs
        return True, None

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs):
        captured["verify_kwargs"] = setup_kwargs
        return True, None

    monkeypatch.setattr(add_module, "_run_download", fake_download)
    monkeypatch.setattr(add_module, "verify_checkpoint", fake_verify)

    rc = cmd_add(_make_args(fake_root, kwarg=["task=omat", "charge=-1"]))
    assert rc == 0
    assert captured["download_kwargs"] == {"task": "omat", "charge": -1}
    assert captured["verify_kwargs"] == {"task": "omat", "charge": -1}


def test_add_invalid_kwarg_returns_2(fake_root):
    rc = cmd_add(_make_args(fake_root, kwarg=["malformed"], no_verify=True))
    assert rc == 2


def test_add_errors_when_env_not_built(tmp_path, monkeypatch):
    # Root has no envs/<x>_env/bin/python anywhere.
    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest

    save_manifest(create_manifest(tmp_path, "test", UserConfig(name="t", email="t@t.t")), tmp_path)

    rc = cmd_add(_make_args(tmp_path, env="nonexistent", no_verify=True))
    assert rc == 1


def test_add_strips_env_suffix(fake_root, monkeypatch):
    """Both 'mace' and 'mace_env' should resolve to the same env."""
    monkeypatch.setattr(add_module, "_run_download", lambda *a, **kw: (True, None))

    rc = cmd_add(_make_args(fake_root, env="mace_env", no_verify=True))
    assert rc == 0

    m = load_manifest(fake_root)
    assert "mace_env" in m.environments
    assert "medium" in m.environments["mace_env"].checkpoints
