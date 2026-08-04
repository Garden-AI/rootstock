"""Tests for ``rootstock add``."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock import operations
from rootstock.commands.add import cmd_add
from rootstock.manifest import load_manifest
from rootstock.operations import parse_kwarg

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


_MACE_ENV_SOURCE = '''\
"""MACE env."""

CHECKPOINTS = {
    "mace-mp-0-small":  "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large":  "large",
}


def setup(checkpoint, device="cuda"):
    return None
'''


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch) -> Path:
    """Build a minimal rootstock root with a fake mace env so cmd_add can run."""
    root = tmp_path
    env_dir = root / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_MACE_ENV_SOURCE)

    # Initialize a manifest so we don't trigger config-loading paths.
    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest

    cfg = UserConfig(name="t", email="t@t.t")
    save_manifest(create_manifest(root, ["test"], cfg), root)

    # Stub update_and_push_manifest — not under test here.
    monkeypatch.setattr(
        "rootstock.operations.update_and_push_manifest",
        lambda *a, **kw: True,
    )
    return root


def _make_args(root: Path, **overrides):
    class _Args:
        pass

    args = _Args()
    args.checkpoint = overrides.get("checkpoint", "mace-mp-0-medium")
    args.list = overrides.get("list", False)
    args.kwarg = overrides.get("kwarg")
    args.device = overrides.get("device", "cuda")
    args.verify_timeout = overrides.get("verify_timeout", 600.0)
    args.no_verify = overrides.get("no_verify", False)
    args.force = overrides.get("force", False)
    args.root = str(root)
    args.no_push = overrides.get("no_push", True)
    args.cluster = overrides.get("cluster")
    return args


def test_add_no_verify_sets_fetched_at(fake_root, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))

    rc = cmd_add(_make_args(fake_root, no_verify=True))
    assert rc == 0

    m = load_manifest(fake_root)
    ckpt = m.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.fetched_at is not None
    assert ckpt.verification("test").verified_at is None
    assert ckpt.last_error is None


def test_add_overrides_restrictive_umask(fake_root, monkeypatch):
    """Weights written to the shared cache must be world-readable regardless
    of the maintainer's personal umask."""
    import os

    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))

    old = os.umask(0o077)
    try:
        assert cmd_add(_make_args(fake_root, no_verify=True)) == 0
        # cmd_add must have replaced the restrictive umask with 002.
        assert os.umask(0o022) == 0o002
    finally:
        os.umask(old)


def test_add_forwards_verify_timeout(fake_root, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))
    captured = {}

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **kwargs):
        captured.update(kwargs)
        return True, None

    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    assert cmd_add(_make_args(fake_root, verify_timeout=1800.0)) == 0
    assert captured["timeout"] == 1800.0


def test_add_then_add_is_idempotent(fake_root, monkeypatch):
    download_calls = []
    verify_calls = []

    def fake_download(*a, **kw):
        download_calls.append((a, kw))
        return True, None

    def fake_verify(*a, **kw):
        verify_calls.append((a, kw))
        return True, None

    monkeypatch.setattr(operations, "_run_download", fake_download)
    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    # First call: --no-verify, only fetches.
    assert cmd_add(_make_args(fake_root, no_verify=True)) == 0
    # Second call: full add; should NOT re-download (idempotent), but should verify.
    assert cmd_add(_make_args(fake_root, no_verify=False)) == 0

    assert len(download_calls) == 1, "download should only happen once"
    assert len(verify_calls) == 1

    m = load_manifest(fake_root)
    ckpt = m.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.fetched_at is not None
    record = ckpt.verification("test")
    assert record.verified_at is not None
    assert record.verified_device == "cuda"
    assert ckpt.last_error is None
    assert record.last_error is None


def test_add_force_redownloads(fake_root, monkeypatch):
    """--force repairs a cache file gone missing behind the manifest's
    fetched stamp."""
    download_calls = []
    monkeypatch.setattr(
        operations,
        "_run_download",
        lambda *a, **kw: (download_calls.append(a), (True, None))[1],
    )

    assert cmd_add(_make_args(fake_root, no_verify=True)) == 0
    assert cmd_add(_make_args(fake_root, no_verify=True, force=True)) == 0

    assert len(download_calls) == 2, "--force must re-run the download"


def test_add_records_download_failure_and_returns_1(fake_root, monkeypatch):
    monkeypatch.setattr(
        operations,
        "_run_download",
        lambda *a, **kw: (False, "ConnectionError: hub unreachable"),
    )
    rc = cmd_add(_make_args(fake_root, no_verify=True))
    assert rc == 1

    m = load_manifest(fake_root)
    ckpt = m.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.fetched_at is None
    assert "ConnectionError" in ckpt.last_error
    assert ckpt.last_error.startswith("download:")


def test_add_records_verify_failure_and_returns_1(fake_root, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))
    monkeypatch.setattr(
        operations,
        "verify_checkpoint",
        lambda *a, **kw: (False, "RuntimeError: CUDA OOM"),
    )

    rc = cmd_add(_make_args(fake_root, no_verify=False))
    assert rc == 1

    m = load_manifest(fake_root)
    ckpt = m.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.fetched_at is not None  # download succeeded, was preserved
    record = ckpt.verification("test")
    assert record.verified_at is None
    assert record.verified_device is None
    assert "CUDA OOM" in record.last_error
    assert record.last_error.startswith("verify:")


def test_add_clears_last_error_on_success(fake_root, monkeypatch):
    """After a verify failure, a successful re-add should clear last_error."""
    # First attempt: verify fails.
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))
    monkeypatch.setattr(
        operations,
        "verify_checkpoint",
        lambda *a, **kw: (False, "ValueError: bad input"),
    )
    cmd_add(_make_args(fake_root, no_verify=False))

    m = load_manifest(fake_root)
    ckpt = m.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.verification("test").last_error is not None

    # Second attempt: verify succeeds. last_error should clear.
    monkeypatch.setattr(operations, "verify_checkpoint", lambda *a, **kw: (True, None))
    rc = cmd_add(_make_args(fake_root, no_verify=False))
    assert rc == 0

    m = load_manifest(fake_root)
    ckpt = m.environments["mace"].checkpoints["mace-mp-0-medium"]
    assert ckpt.verification("test").last_error is None
    assert ckpt.last_error is None


def test_add_forwards_kwargs_to_download_and_verify(fake_root, monkeypatch):
    captured = {}

    def fake_download(root, env_name, checkpoint, setup_kwargs, **_):
        captured["download_kwargs"] = setup_kwargs
        return True, None

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **_):
        captured["verify_kwargs"] = setup_kwargs
        return True, None

    monkeypatch.setattr(operations, "_run_download", fake_download)
    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    rc = cmd_add(_make_args(fake_root, kwarg=["task=omat", "charge=-1"]))
    assert rc == 0
    assert captured["download_kwargs"] == {"task": "omat", "charge": -1}
    assert captured["verify_kwargs"] == {"task": "omat", "charge": -1}


def test_add_invalid_kwarg_returns_2(fake_root):
    rc = cmd_add(_make_args(fake_root, kwarg=["malformed"], no_verify=True))
    assert rc == 2


def test_add_errors_on_unknown_checkpoint(fake_root):
    """An id no installed env declares should fail with an informative error."""
    rc = cmd_add(_make_args(fake_root, checkpoint="not-a-real-id", no_verify=True))
    assert rc == 1


def test_add_errors_when_no_envs_installed(tmp_path):
    """No installed envs anywhere: the error message points at install."""
    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest

    save_manifest(
        create_manifest(tmp_path, ["test"], UserConfig(name="t", email="t@t.t")), tmp_path
    )

    rc = cmd_add(_make_args(tmp_path, checkpoint="mace-mp-0-medium", no_verify=True))
    assert rc == 1


# ---------- cmd_add --list (checkpoint catalog) ---------------------------


def test_add_list_shows_declared_checkpoints(fake_root, capsys):
    """--list prints every declared id grouped by env, without touching the
    download/verify paths."""
    rc = cmd_add(_make_args(fake_root, list=True, checkpoint=None))
    assert rc == 0

    out = capsys.readouterr().out
    assert "mace:" in out
    for ckpt_id in ("mace-mp-0-small", "mace-mp-0-medium", "mace-mp-0-large"):
        assert ckpt_id in out


def test_add_list_when_no_envs_installed(tmp_path, capsys):
    """--list on an empty root points the user at install rather than erroring."""
    rc = cmd_add(_make_args(tmp_path, list=True, checkpoint=None))
    assert rc == 0

    out = capsys.readouterr().out
    assert "No envs are installed" in out
    assert "rootstock install" in out


def test_add_without_checkpoint_or_list_returns_2(fake_root):
    """Omitting both the checkpoint and --list is a usage error."""
    rc = cmd_add(_make_args(fake_root, checkpoint=None))
    assert rc == 2


# ---------- ':custom' checkpoints in cmd_add ---------------------------------


def test_add_of_custom_id_points_at_weights(fake_root, capsys):
    """`rootstock add <family>:custom` is a category error — nothing to
    download or register; point at direct use with a weights file."""
    rc = cmd_add(_make_args(fake_root, checkpoint="mace-mp:custom"))
    assert rc == 2
    err = capsys.readouterr().err
    assert "nothing to add" in err
    assert "weights" in err
