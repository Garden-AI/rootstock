"""Node-local staging hygiene: multi-user directory permissions, eviction's
live-worker shield, and remapping through install-time mount-alias
spellings."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from rootstock.stage import (
    _evict_lru,
    _mark_in_use,
    _remap_into_stage,
    _user_stage_root,
)

SEVEN_HOURS_AGO = time.time() - 7 * 3600


def _age(path: Path) -> None:
    os.utime(path, (SEVEN_HOURS_AGO, SEVEN_HOURS_AGO))


def test_shared_intermediate_is_sticky_world_writable(tmp_path: Path):
    """{base}/rootstock is created by whoever stages first; under their
    umask it must still let every other user create a leaf — the /tmp
    recipe (sticky 1777), with the per-user leaf locked to 0700."""
    old_umask = os.umask(0o077)
    try:
        user_root = _user_stage_root(tmp_path)
    finally:
        os.umask(old_umask)
    shared = tmp_path / "rootstock"
    assert shared.stat().st_mode & 0o7777 == 0o1777
    assert user_root.stat().st_mode & 0o7777 == 0o700


def test_eviction_spares_envs_with_live_users(tmp_path: Path):
    """Dir mtime alone can't shield a multi-day MD run: a staged env with a
    live registered client pid must survive eviction however old it is."""
    envs_root = tmp_path / "envs-by-hash"
    envs_root.mkdir()
    keep = envs_root / "keep"
    keep.mkdir()

    in_use = envs_root / "sha-in-use"
    in_use.mkdir()
    _mark_in_use(in_use)  # registers our own (alive) pid
    _age(in_use)

    import subprocess

    reaped = subprocess.Popen(["true"])
    reaped.wait()
    dead = envs_root / "sha-dead"
    (dead / ".users").mkdir(parents=True)
    (dead / ".users" / str(reaped.pid)).touch()
    _age(dead)

    # An unreachable target forces the scan over every candidate.
    unreachable = shutil.disk_usage(envs_root).free + 10**15
    _evict_lru(envs_root, keep=keep, bytes_needed=unreachable)

    assert in_use.exists()  # live client: shielded
    assert not dead.exists()  # dead pidfile: evicted
    assert keep.exists()


def test_remap_resolves_install_time_alias_spelling(tmp_path: Path):
    """uv bakes the install-time path spelling into symlink targets and
    pyvenv.cfg; on multi-alias mounts (/eagle vs /lus/eagle) that spelling
    matches neither the spawn-time root nor its resolution — the value
    itself must be resolved and retried."""
    real_root = tmp_path / "lus" / "eagle" / "rootstock"
    (real_root / ".python").mkdir(parents=True)
    (tmp_path / "eagle").symlink_to(tmp_path / "lus" / "eagle")
    alias_target = str(tmp_path / "eagle" / "rootstock" / ".python" / "cp311" / "bin" / "python")
    staged = tmp_path / "staged"

    remapped = _remap_into_stage(alias_target, real_root, staged)

    assert remapped == str(staged / ".python" / "cp311" / "bin" / "python")


def test_remap_leaves_foreign_paths_alone(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    assert _remap_into_stage("/usr/bin/python3", root, tmp_path / "staged") is None
    assert _remap_into_stage("3.11.99", root, tmp_path / "staged") is None
