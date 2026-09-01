"""The node-local weight mirror (#180): recorded files materialize locally,
everything else falls through to the shared cache via symlinks — including
the HuggingFace-hub snapshot indirection, whose relative symlinks must
resolve to the *local* blob copies."""

from __future__ import annotations

import os
from pathlib import Path

from stagelib import ENV_NAME, write_manifest_env

from rootstock.stage import stage_weights

CKPT = "demo-checkpoint"


def _record(root: Path, entries: list[dict]) -> None:
    write_manifest_env(root, checkpoints={CKPT: {"weight_files": entries}})


def _write(base: Path, rel: str, data: bytes) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_recorded_files_copy_and_siblings_symlink(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    weights = _write(root, "cache/demo/weights.pt", b"w" * 512)
    config = _write(root, "cache/demo/config.json", b"{}")
    _write(root, "cache/other-family/big.bin", b"b" * 128)
    _record(root, [{"path": "cache/demo/weights.pt", "size": 512}])
    base = tmp_path / "local"
    base.mkdir()

    result = stage_weights(root, None, ENV_NAME, CKPT, base)

    assert result is not None
    mirror, copied = result
    assert copied == 512
    staged = mirror / "cache" / "demo" / "weights.pt"
    assert staged.is_file() and not staged.is_symlink()
    assert staged.read_bytes() == weights.read_bytes()
    # Unrecorded sibling falls through to the shared copy…
    linked_config = mirror / "cache" / "demo" / "config.json"
    assert linked_config.is_symlink() and linked_config.resolve() == config.resolve()
    # …and untouched families are one whole-directory symlink, not a walk.
    other = mirror / "cache" / "other-family"
    assert other.is_symlink()
    # The worker's HOME must exist locally even with no recorded files there.
    assert (mirror / "home").is_dir() and not (mirror / "home").is_symlink()


def test_second_pass_skips_current_copies(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    _write(root, "cache/demo/weights.pt", b"w" * 512)
    _record(root, [{"path": "cache/demo/weights.pt", "size": 512}])
    base = tmp_path / "local"
    base.mkdir()

    first = stage_weights(root, None, ENV_NAME, CKPT, base)
    second = stage_weights(root, None, ENV_NAME, CKPT, base)

    assert first is not None and second is not None
    assert first[1] == 512 and second[1] == 0  # rsync-style: no re-copy


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

    result = stage_weights(root, None, ENV_NAME, CKPT, base)

    assert result is not None
    mirror, _ = result
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

    result = stage_weights(root, cache_root, ENV_NAME, CKPT, base)

    assert result is not None
    mirror, copied = result
    assert copied == 64
    assert (mirror / "cache" / "demo" / "weights.pt").read_bytes() == b"w" * 64
