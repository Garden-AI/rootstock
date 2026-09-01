"""The fetch/verify split of ``add_checkpoint``.

``fetch_checkpoint`` (CPU download, freely parallel) and
``verify_fetched_checkpoint`` (GPU model load, bounded) are usable on their
own so a batch driver can run the two phases at different times, in different
jobs, with different concurrency. ``add_checkpoint`` is their composition and
must behave exactly as before the split — including refreshing the manifest
once, at the end, and not at all on failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock import operations
from rootstock.manifest import load_manifest
from rootstock.operations import (
    OperationError,
    add_checkpoint,
    fetch_checkpoint,
    verify_fetched_checkpoint,
)

_MACE_ENV_SOURCE = '''\
"""MACE env."""

CHECKPOINTS = {
    "mace-mp-0-medium": "medium",
}


def setup(checkpoint, device="cuda"):
    return None
'''


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A minimal rootstock root with a fake mace env and a manifest."""
    root = tmp_path
    env_dir = root / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_MACE_ENV_SOURCE)

    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest

    cfg = UserConfig(name="t", email="t@t.t")
    save_manifest(create_manifest(root, ["test"], cfg), root)
    return root


@pytest.fixture
def refresh_calls(monkeypatch) -> list:
    """Stub update_and_push_manifest, recording each call."""
    calls: list = []

    def fake_refresh(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(operations, "update_and_push_manifest", fake_refresh)
    return calls


def _ckpt(root: Path):
    return load_manifest(root).environments["mace"].checkpoints["mace-mp-0-medium"]


# ---------- fetch_checkpoint ------------------------------------------------


def test_fetch_records_fetched_at(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))

    result = fetch_checkpoint(fake_root, "mace-mp-0-medium")

    assert result.env_name == "mace"
    assert result.fetched_at is not None
    assert not result.already_fetched
    ckpt = _ckpt(fake_root)
    assert ckpt.fetched_at == result.fetched_at
    assert ckpt.verification("test").verified_at is None
    assert ckpt.last_error is None


def test_fetch_is_idempotent(fake_root, refresh_calls, monkeypatch):
    downloads = []
    monkeypatch.setattr(
        operations, "_run_download", lambda *a, **kw: (downloads.append(a), (True, None))[1]
    )

    first = fetch_checkpoint(fake_root, "mace-mp-0-medium")
    second = fetch_checkpoint(fake_root, "mace-mp-0-medium")

    assert len(downloads) == 1, "download should only happen once"
    assert second.already_fetched
    assert second.fetched_at == first.fetched_at


def test_fetch_force_redownloads_past_fetched_stamp(fake_root, refresh_calls, monkeypatch):
    """The repair path for cache files gone missing behind the manifest."""
    downloads = []
    monkeypatch.setattr(
        operations, "_run_download", lambda *a, **kw: (downloads.append(a), (True, None))[1]
    )

    first = fetch_checkpoint(fake_root, "mace-mp-0-medium")
    second = fetch_checkpoint(fake_root, "mace-mp-0-medium", force=True)

    assert len(downloads) == 2, "force must re-run the download"
    assert not second.already_fetched
    assert second.fetched_at is not None
    assert second.fetched_at >= first.fetched_at


def test_fetch_failure_records_last_error_and_raises(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(
        operations, "_run_download", lambda *a, **kw: (False, "ConnectionError: hub unreachable")
    )

    with pytest.raises(OperationError, match="download failed"):
        fetch_checkpoint(fake_root, "mace-mp-0-medium")

    ckpt = _ckpt(fake_root)
    assert ckpt.fetched_at is None
    assert ckpt.last_error.startswith("download:")


def test_fetch_refresh_knob(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))

    fetch_checkpoint(fake_root, "mace-mp-0-medium", refresh=False)
    assert refresh_calls == [], "refresh=False must skip update_and_push_manifest"

    fetch_checkpoint(fake_root, "mace-mp-0-medium")
    assert len(refresh_calls) == 1


_MULTIHEAD_ENV_SOURCE = '''\
"""Multi-head env: setup() requires a head selection, no default."""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
}

VERIFY_KWARGS = {
    "uma-s-1p1": {"task": "omat"},
}


def setup(checkpoint, device="cuda", task=None):
    return None
'''


@pytest.fixture
def multihead_root(tmp_path: Path) -> Path:
    """A root whose env declares VERIFY_KWARGS for its checkpoint."""
    root = tmp_path
    env_dir = root / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_MULTIHEAD_ENV_SOURCE)

    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest

    cfg = UserConfig(name="t", email="t@t.t")
    save_manifest(create_manifest(root, ["test"], cfg), root)
    return root


def test_fetch_falls_back_to_verify_kwargs(multihead_root, refresh_calls, monkeypatch):
    """No explicit kwargs → the env's VERIFY_KWARGS reach the download's
    setup() call, same as the verify path (a multi-head setup() raises
    without a head selection)."""
    captured = {}

    def fake_download(root, env_name, checkpoint, setup_kwargs, **kw):
        captured["setup_kwargs"] = setup_kwargs
        return True, None

    monkeypatch.setattr(operations, "_run_download", fake_download)

    fetch_checkpoint(multihead_root, "uma-s-1p1")

    assert captured["setup_kwargs"] == {"task": "omat"}


def test_fetch_explicit_kwargs_beat_verify_kwargs(multihead_root, refresh_calls, monkeypatch):
    captured = {}

    def fake_download(root, env_name, checkpoint, setup_kwargs, **kw):
        captured["setup_kwargs"] = setup_kwargs
        return True, None

    monkeypatch.setattr(operations, "_run_download", fake_download)

    fetch_checkpoint(multihead_root, "uma-s-1p1", setup_kwargs={"task": "omol"})

    assert captured["setup_kwargs"] == {"task": "omol"}


def test_fetch_fails_fast_when_env_not_built(tmp_path, refresh_calls):
    from rootstock.config import UserConfig
    from rootstock.manifest import create_manifest, save_manifest

    save_manifest(
        create_manifest(tmp_path, ["test"], UserConfig(name="t", email="t@t.t")), tmp_path
    )
    env_dir = tmp_path / "envs" / "mace"
    env_dir.mkdir(parents=True)
    (env_dir / "env_source.py").write_text(_MACE_ENV_SOURCE)  # declared but not built

    with pytest.raises(OperationError, match="not built"):
        fetch_checkpoint(tmp_path, "mace-mp-0-medium")


# ---------- verify_fetched_checkpoint ----------------------------------------


def test_verify_records_outcome(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(operations, "verify_checkpoint", lambda *a, **kw: (True, None))

    result = verify_fetched_checkpoint(fake_root, "mace-mp-0-medium", device="cuda")

    assert result.env_name == "mace"
    assert result.verified_device == "cuda"
    record = _ckpt(fake_root).verification("test")
    assert record.verified_at == result.verified_at
    assert record.verified_device == "cuda"
    assert record.last_error is None


def test_verify_failure_clears_stamps_and_raises(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(
        operations, "verify_checkpoint", lambda *a, **kw: (False, "RuntimeError: CUDA OOM")
    )

    with pytest.raises(OperationError, match="verify failed"):
        verify_fetched_checkpoint(fake_root, "mace-mp-0-medium")

    record = _ckpt(fake_root).verification("test")
    assert record.verified_at is None
    assert record.verified_device is None
    assert "CUDA OOM" in record.last_error
    assert record.last_error.startswith("verify:")


def test_verify_refresh_knob(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(operations, "verify_checkpoint", lambda *a, **kw: (True, None))

    verify_fetched_checkpoint(fake_root, "mace-mp-0-medium", refresh=False)
    assert refresh_calls == [], "refresh=False must skip update_and_push_manifest"

    verify_fetched_checkpoint(fake_root, "mace-mp-0-medium")
    assert len(refresh_calls) == 1


def test_verify_forwards_device_and_kwargs(fake_root, refresh_calls, monkeypatch):
    captured = {}

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **_):
        captured["device"] = device
        captured["setup_kwargs"] = setup_kwargs
        return True, None

    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    verify_fetched_checkpoint(
        fake_root, "mace-mp-0-medium", device="cuda:1", setup_kwargs={"task": "omat"}
    )

    assert captured["device"] == "cuda:1"
    assert captured["setup_kwargs"] == {"task": "omat"}
    assert _ckpt(fake_root).verification("test").verified_device == "cuda:1"


def test_verify_forwards_timeout(fake_root, refresh_calls, monkeypatch):
    captured = {}

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **kwargs):
        captured.update(kwargs)
        return True, None

    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    verify_fetched_checkpoint(fake_root, "mace-mp-0-medium", timeout=1800.0)
    assert captured["timeout"] == 1800.0

    verify_fetched_checkpoint(fake_root, "mace-mp-0-medium")
    assert captured["timeout"] == 600.0, "default must stay at 600s"


# ---------- add_checkpoint (the composition) ---------------------------------


def test_add_refreshes_manifest_exactly_once(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))
    monkeypatch.setattr(operations, "verify_checkpoint", lambda *a, **kw: (True, None))

    result = add_checkpoint(fake_root, "mace-mp-0-medium")

    assert len(refresh_calls) == 1, "add must refresh once, at the end"
    assert result.fetched_at is not None
    assert result.verified_at is not None
    assert result.verified_device == "cuda"


def test_add_no_verify_skips_verify_fields(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))

    result = add_checkpoint(fake_root, "mace-mp-0-medium", verify=False)

    assert result.verified_at is None
    assert result.verified_device is None
    assert len(refresh_calls) == 1


def test_add_forwards_verify_timeout(fake_root, refresh_calls, monkeypatch):
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))
    captured = {}

    def fake_verify(root, env_name, checkpoint, device, setup_kwargs, **kwargs):
        captured.update(kwargs)
        return True, None

    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    add_checkpoint(fake_root, "mace-mp-0-medium", verify_timeout=1800.0)
    assert captured["timeout"] == 1800.0


def test_add_failure_skips_the_trailing_refresh(fake_root, refresh_calls, monkeypatch):
    """Pre-split behavior: a failed add records last_error but never reaches
    the final refresh."""
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (False, "boom"))

    with pytest.raises(OperationError):
        add_checkpoint(fake_root, "mace-mp-0-medium")

    assert refresh_calls == []
