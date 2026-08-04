"""Manifest weight records from capture results (issue #177).

The capture subprocess writes a JSON result; the operations layer turns it
into ``CheckpointInfo.weight_files`` under a deliberate merge policy: a
non-trivial capture replaces the record (freshness self-heals on every
add/verify/smoke-test pass), a suspiciously small one keeps the previous
record, and "no capture ran" learns nothing — it must never erase what a
healthier run recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock import operations
from rootstock.manifest import CheckpointInfo, load_manifest
from rootstock.operations import (
    OperationError,
    apply_weights_record,
    fetch_checkpoint,
    read_weights_capture,
    verify_fetched_checkpoint,
)

_FILES = [
    {"path": "cache/fake/model.bin", "size": 9_000_000},
    {"path": "home/.cache/fairchem/uma.pt", "size": 5_000_000},
]

_MACE_ENV_SOURCE = '''\
"""MACE env."""

CHECKPOINTS = {
    "mace-mp-0-medium": "medium",
}


def setup(checkpoint, device="cuda"):
    return None
'''


# ---- read_weights_capture ----------------------------------------------------


def test_read_missing_file_is_none(tmp_path):
    assert read_weights_capture(tmp_path / "nope.json") is None


@pytest.mark.parametrize(
    "content",
    [
        "not json {",
        json.dumps({"nope": []}),
        json.dumps({"files": [{"path": 42, "size": 1}]}),
        json.dumps({"files": [{"path": "a", "size": "big"}]}),
    ],
)
def test_read_unusable_capture_is_none(tmp_path, content):
    path = tmp_path / "weights.json"
    path.write_text(content)
    assert read_weights_capture(path) is None


def test_read_valid_capture(tmp_path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"files": _FILES}))
    assert read_weights_capture(path) == _FILES


# ---- apply_weights_record merge policy ----------------------------------------


def test_nontrivial_capture_replaces_record_sorted():
    ckpt = CheckpointInfo(
        weight_files=[{"path": "cache/old.bin", "size": 2_000_000}],
        weights_recorded_at="2026-01-01T00:00:00+00:00",
    )
    apply_weights_record(ckpt, list(reversed(_FILES)))
    assert ckpt.weight_files == _FILES  # sorted by path
    assert ckpt.weights_recorded_at > "2026-01-01"


def test_none_capture_changes_nothing():
    old = [{"path": "cache/old.bin", "size": 2_000_000}]
    ckpt = CheckpointInfo(weight_files=old, weights_recorded_at="2026-01-01T00:00:00+00:00")
    apply_weights_record(ckpt, None)
    assert ckpt.weight_files == old
    assert ckpt.weights_recorded_at == "2026-01-01T00:00:00+00:00"


def test_tiny_capture_keeps_old_record_and_says_so():
    """Below the byte floor the capture is presumed a broken probe, not a
    real working set — replacing a good record with it would strand the
    prewarm on its heuristic fallback."""
    old = [{"path": "cache/old.bin", "size": 2_000_000}]
    ckpt = CheckpointInfo(weight_files=old, weights_recorded_at="2026-01-01T00:00:00+00:00")
    said: list[str] = []
    apply_weights_record(
        ckpt, [{"path": "cache/crumb", "size": 12}], label="mace/x", progress=said.append
    )
    assert ckpt.weight_files == old
    assert any("suspiciously small" in line for line in said)


def test_tiny_capture_with_no_old_record_stays_unrecorded():
    """An empty-ish record must read as 'never recorded' (heuristic tier),
    not 'recorded: warm nothing'."""
    ckpt = CheckpointInfo()
    apply_weights_record(ckpt, [])
    assert ckpt.weight_files is None
    assert ckpt.weights_recorded_at is None


# ---- fetch/verify plumbing ----------------------------------------------------


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
def no_refresh(monkeypatch):
    monkeypatch.setattr(operations, "update_and_push_manifest", lambda *a, **kw: True)


def _ckpt(root: Path):
    return load_manifest(root).environments["mace"].checkpoints["mace-mp-0-medium"]


def test_fetch_records_weight_files(fake_root, no_refresh, monkeypatch):
    """fetch passes a capture path to the download subprocess and merges
    whatever the wrapper wrote there into the manifest record."""

    def fake_download(
        root, env_name, checkpoint, setup_kwargs, cache_root=None, weights_capture_path=None
    ):
        Path(weights_capture_path).write_text(json.dumps({"files": _FILES}))
        return True, None

    monkeypatch.setattr(operations, "_run_download", fake_download)

    fetch_checkpoint(fake_root, "mace-mp-0-medium")

    ckpt = _ckpt(fake_root)
    assert ckpt.weight_files == _FILES
    assert ckpt.weights_recorded_at is not None


def test_fetch_without_capture_leaves_record_alone(fake_root, no_refresh, monkeypatch):
    """A wrapper that wrote nothing (old staged helper, capture error) must
    not disturb an existing record."""
    monkeypatch.setattr(operations, "_run_download", lambda *a, **kw: (True, None))

    fetch_checkpoint(fake_root, "mace-mp-0-medium")

    ckpt = _ckpt(fake_root)
    assert ckpt.fetched_at is not None
    assert ckpt.weight_files is None


def test_verify_records_weight_files_even_on_failure(fake_root, no_refresh, monkeypatch):
    """The capture only exists if setup() completed; a forward pass that then
    fails (CUDA OOM, bad forces) doesn't invalidate the load working set."""

    def fake_verify(
        root,
        env_name,
        checkpoint,
        device,
        setup_kwargs,
        *,
        cache_root=None,
        weights_capture_path=None,
        **_,
    ):
        Path(weights_capture_path).write_text(json.dumps({"files": _FILES}))
        return False, "forces are all (near-)zero"

    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    with pytest.raises(OperationError, match="verify failed"):
        verify_fetched_checkpoint(fake_root, "mace-mp-0-medium")

    ckpt = _ckpt(fake_root)
    assert ckpt.verification("test").verified_at is None
    assert ckpt.weight_files == _FILES


def test_verify_success_records_weight_files(fake_root, no_refresh, monkeypatch):
    def fake_verify(
        root,
        env_name,
        checkpoint,
        device,
        setup_kwargs,
        *,
        cache_root=None,
        weights_capture_path=None,
        **_,
    ):
        Path(weights_capture_path).write_text(json.dumps({"files": _FILES}))
        return True, None

    monkeypatch.setattr(operations, "verify_checkpoint", fake_verify)

    verify_fetched_checkpoint(fake_root, "mace-mp-0-medium")

    ckpt = _ckpt(fake_root)
    assert ckpt.verification("test").verified_at is not None
    assert ckpt.weight_files == _FILES
