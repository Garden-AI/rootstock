"""Tests for the manifest write lock (O_EXCL lock file — never flock; see #125)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from rootstock.manifest import ManifestError, ManifestLockTimeout, manifest_lock


def test_lock_file_created_and_removed(tmp_path: Path):
    lock = tmp_path / "manifest.json.lock"
    with manifest_lock(tmp_path):
        assert lock.exists()
        assert json.loads(lock.read_text())["pid"] == os.getpid()
    assert not lock.exists()


def test_lock_removed_on_exception(tmp_path: Path):
    with pytest.raises(RuntimeError, match="boom"):
        with manifest_lock(tmp_path):
            raise RuntimeError("boom")
    assert not (tmp_path / "manifest.json.lock").exists()


def test_lock_creates_missing_root(tmp_path: Path):
    root = tmp_path / "not" / "yet" / "there"
    with manifest_lock(root):
        pass
    assert root.is_dir()


def test_contended_lock_times_out_cleanly(tmp_path: Path):
    with manifest_lock(tmp_path):
        with pytest.raises(ManifestLockTimeout, match="manifest.json"):
            with manifest_lock(tmp_path, timeout=1.0):
                pass
    # The loser must not have removed the winner's lock on its way out —
    # but the winner's own exit above does, so re-check the release worked.
    assert not (tmp_path / "manifest.json.lock").exists()


def test_lock_timeout_is_a_manifest_error():
    """The CLI catches ManifestError for a clean diagnosis — lock timeouts
    must ride that path, not escape as tracebacks."""
    assert issubclass(ManifestLockTimeout, ManifestError)


def test_timeout_message_names_the_holder(tmp_path: Path):
    lock = tmp_path / "manifest.json.lock"
    lock.write_text('{"pid": 4242, "host": "nid001", "created": "2026-07-17T00:00:00Z"}')
    with pytest.raises(ManifestLockTimeout, match="4242"):
        with manifest_lock(tmp_path, timeout=0.6):
            pass
    assert lock.exists()  # a fresh foreign lock is respected, not broken


def test_stale_lock_is_broken(tmp_path: Path):
    lock = tmp_path / "manifest.json.lock"
    lock.write_text('{"pid": 999999, "host": "dead-node", "created": "old"}')
    hour_ago = time.time() - 3600
    os.utime(lock, (hour_ago, hour_ago))

    with manifest_lock(tmp_path, timeout=5.0, stale_after=600.0):
        # We hold our own lock now, not the corpse.
        assert json.loads(lock.read_text())["pid"] == os.getpid()
    assert not lock.exists()


def test_transactions_load_fresh_and_do_not_lose_updates(tmp_path: Path):
    """Two sequential read-modify-write cycles each see the other's writes —
    the lost-update mode was mutating a long-held manifest object and saving
    it over someone else's interleaved save."""
    from rootstock.manifest import CheckpointInfo, EnvironmentInfo, load_manifest
    from rootstock.operations import _manifest_transaction

    def env_record() -> EnvironmentInfo:
        return EnvironmentInfo(
            built_at="2026-01-01T00:00:00Z",
            source_hash="sha256:abc",
            source="",
            python_requires=">=3.11",
            dependencies={},
            checkpoints={"ck": CheckpointInfo()},
        )

    with _manifest_transaction(tmp_path, cluster_hint="test") as manifest:
        manifest.environments["a"] = env_record()

    with _manifest_transaction(tmp_path, cluster_hint="test") as manifest:
        # Loaded fresh: writer A's env is visible, not clobbered.
        assert "a" in manifest.environments
        manifest.environments["b"] = env_record()

    final = load_manifest(tmp_path)
    assert set(final.environments) == {"a", "b"}


def test_transaction_does_not_save_on_exception(tmp_path: Path):
    from rootstock.manifest import load_manifest
    from rootstock.operations import _manifest_transaction

    with pytest.raises(RuntimeError, match="boom"):
        with _manifest_transaction(tmp_path, cluster_hint="test"):
            raise RuntimeError("boom")
    assert load_manifest(tmp_path) is None
    assert not (tmp_path / "manifest.json.lock").exists()
