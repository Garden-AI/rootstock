"""The node-local weight mirror (#180): recorded files materialize locally,
everything else falls through to the shared cache via symlinks — including
the HuggingFace-hub snapshot indirection, whose relative symlinks must
resolve to the *local* blob copies. Currency is size+mtime, and a completed
overlay leaves a per-checkpoint marker that later spawns re-enter lock-free.
"""

from __future__ import annotations

import os
from pathlib import Path

from stagelib import ENV_NAME, write_manifest_env

import rootstock.stage as stage_module
from rootstock.stage import stage_weights

CKPT = "demo-checkpoint"


def _record(root: Path, entries: list[dict]) -> None:
    write_manifest_env(root, checkpoints={CKPT: {"weight_files": entries}})


def _write(base: Path, rel: str, data: bytes) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _bump_mtime(path: Path) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def test_recorded_files_copy_and_siblings_symlink(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    weights = _write(root, "cache/demo/weights.pt", b"w" * 512)
    config = _write(root, "cache/demo/config.json", b"{}")
    _write(root, "cache/other-family/big.bin", b"b" * 128)
    _record(root, [{"path": "cache/demo/weights.pt", "size": 512}])
    base = tmp_path / "local"
    base.mkdir()

    mirror = stage_weights(root, None, ENV_NAME, CKPT, base)

    assert mirror is not None
    staged = mirror / "cache" / "demo" / "weights.pt"
    assert staged.is_file() and not staged.is_symlink()
    assert staged.read_bytes() == weights.read_bytes()
    # The copy carries the source's mtime — that is the staleness signal.
    assert staged.stat().st_mtime_ns == weights.stat().st_mtime_ns
    # Unrecorded sibling falls through to the shared copy…
    linked_config = mirror / "cache" / "demo" / "config.json"
    assert linked_config.is_symlink() and linked_config.resolve() == config.resolve()
    # …and untouched families are one whole-directory symlink, not a walk.
    other = mirror / "cache" / "other-family"
    assert other.is_symlink()
    # The worker's HOME must exist locally even with no recorded files there.
    assert (mirror / "home").is_dir() and not (mirror / "home").is_symlink()


def test_second_pass_is_lock_free_and_copies_nothing(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _write(root, "cache/demo/weights.pt", b"w" * 512)
    _record(root, [{"path": "cache/demo/weights.pt", "size": 512}])
    base = tmp_path / "local"
    base.mkdir()

    first = stage_weights(root, None, ENV_NAME, CKPT, base)
    assert first is not None
    inode = (first / "cache" / "demo" / "weights.pt").stat().st_ino

    # The completed-overlay marker means the warm path never takes the
    # mirror lock (the committee-demo serialization finding).
    def no_lock(*a, **k):  # pragma: no cover - the assertion is that it's unused
        raise AssertionError("warm mirror must not take the lock")

    monkeypatch.setattr(stage_module._StageLock, "try_acquire", no_lock)
    second = stage_weights(root, None, ENV_NAME, CKPT, base)

    assert second == first
    # No re-copy: the atomic copy replaces inodes, so an unchanged inode
    # proves nothing was rewritten.
    assert (second / "cache" / "demo" / "weights.pt").stat().st_ino == inode


def test_same_size_source_update_invalidates_mirror(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    src = _write(root, "cache/demo/weights.pt", b"v1" * 256)
    _record(root, [{"path": "cache/demo/weights.pt", "size": 512}])
    base = tmp_path / "local"
    base.mkdir()

    assert stage_weights(root, None, ENV_NAME, CKPT, base) is not None

    # In-place retrain: same size, different bytes, newer mtime. Size-only
    # currency would serve stale forces from the warm mirror forever.
    src.write_bytes(b"v2" * 256)
    _bump_mtime(src)

    mirror = stage_weights(root, None, ENV_NAME, CKPT, base)

    assert mirror is not None
    assert (mirror / "cache" / "demo" / "weights.pt").read_bytes() == b"v2" * 256


def test_hub_snapshot_symlinks_resolve_to_local_blobs(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    repo = "cache/huggingface/hub/models--facebook--UMA"
    blob = _write(root, f"{repo}/blobs/abc123", b"weights" * 100)
    snap_dir = root / repo / "snapshots" / "rev1"
    snap_dir.mkdir(parents=True)
    os.symlink("../../blobs/abc123", snap_dir / "model.safetensors")
    _write(root, f"{repo}/refs/main", b"rev1")
    _record(root, [{"path": f"{repo}/blobs/abc123", "size": blob.stat().st_size}])
    base = tmp_path / "local"
    base.mkdir()

    mirror = stage_weights(root, None, ENV_NAME, CKPT, base)

    assert mirror is not None
    local_blob = mirror / repo / "blobs" / "abc123"
    assert local_blob.is_file() and not local_blob.is_symlink()
    # The snapshot's relative symlink must land on the LOCAL blob — if
    # snapshots/ were a whole-directory symlink into the shared tree, the
    # worker would silently mmap the shared copy and staging would be moot.
    snapshot = mirror / repo / "snapshots" / "rev1" / "model.safetensors"
    assert snapshot.is_symlink()
    assert snapshot.resolve() == local_blob.resolve()
    assert (mirror / repo / "refs" / "main").read_bytes() == b"rev1"


def test_no_record_skips_overlay(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    _record(root, [])
    base = tmp_path / "local"
    base.mkdir()
    assert stage_weights(root, None, ENV_NAME, CKPT, base) is None


def test_purged_recorded_file_skips_overlay(tmp_path: Path):
    # A record whose file was scratch-swept is stale; redirecting caches at
    # a mirror that can't materialize it would break the worker offline.
    root = tmp_path / "root"
    root.mkdir()
    _record(root, [{"path": "cache/demo/gone.pt", "size": 10}])
    base = tmp_path / "local"
    base.mkdir()
    assert stage_weights(root, None, ENV_NAME, CKPT, base) is None


def test_split_cache_root(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    cache_root = tmp_path / "scratch-cache"
    _write(cache_root, "cache/demo/weights.pt", b"w" * 64)
    _record(root, [{"path": "cache/demo/weights.pt", "size": 64}])
    base = tmp_path / "local"
    base.mkdir()

    mirror = stage_weights(root, cache_root, ENV_NAME, CKPT, base)

    assert mirror is not None
    assert (mirror / "cache" / "demo" / "weights.pt").read_bytes() == b"w" * 64


def test_warm_mirror_with_full_disk_stays_staged(tmp_path: Path, monkeypatch):
    """The free-space gate counts only bytes that still need copying — a
    warm mirror on a full disk must not get demoted to the shared path."""
    root = tmp_path / "root"
    root.mkdir()
    _write(root, "cache/demo/weights.pt", b"w" * 512)
    _record(root, [{"path": "cache/demo/weights.pt", "size": 512}])
    base = tmp_path / "local"
    base.mkdir()
    first = stage_weights(root, None, ENV_NAME, CKPT, base)
    assert first is not None

    # Wipe the marker so the pass re-evaluates copies (not the fast path),
    # then report a full disk: nothing needs copying, so it must succeed.
    for marker in (base / "rootstock").rglob("cache-mirror.*.ok"):
        marker.unlink()
    import shutil as _shutil

    usage = _shutil.disk_usage(base)
    monkeypatch.setattr("rootstock.stage.shutil.disk_usage", lambda p: usage._replace(free=0))

    assert stage_weights(root, None, ENV_NAME, CKPT, base) == first
