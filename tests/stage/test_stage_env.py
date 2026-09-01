"""Extracting a packed env image to a node-local dir (#180): the content-
addressed round trip, the venv fixups, warm reuse, and every fall-back-to-
prewarm edge (stale image, missing archive, no record)."""

from __future__ import annotations

import os
from pathlib import Path

from stagelib import ENV_NAME, INTERP, requires_archive_tools, write_manifest_env

from rootstock.pack import pack_environment
from rootstock.stage import stage_env

pytestmark = requires_archive_tools


def _pack_and_record(install_root: Path, packed_at="2026-09-01T00:00:01Z", **kwargs) -> dict:
    record = pack_environment(install_root, ENV_NAME)
    record["packed_at"] = packed_at
    write_manifest_env(install_root, image=record, **kwargs)
    return record


def test_stage_extracts_and_localizes(install_root: Path, tmp_path: Path):
    record = _pack_and_record(install_root)
    base = tmp_path / "local"
    base.mkdir()

    staged_root = stage_env(install_root, ENV_NAME, base)

    assert staged_root is not None
    assert staged_root.name == record["sha256"]
    env_dir = staged_root / "envs" / ENV_NAME

    # The venv now runs on the *staged* interpreter, not the shared one:
    python = env_dir / "bin" / "python"
    assert os.readlink(python) == str(staged_root / ".python" / INTERP / "bin" / "python3.11")
    assert python.exists()
    # relative sibling symlinks survive untouched
    assert os.readlink(env_dir / "bin" / "python3") == "python"
    # pyvenv.cfg's home (where the stdlib is resolved from) is local too
    assert (
        f"home = {staged_root / '.python' / INTERP / 'bin'}" in (env_dir / "pyvenv.cfg").read_text()
    )
    # payload files came through
    assert (env_dir / "lib" / "python3.11" / "site-packages" / "libdemo.so").is_file()
    # no lock/partial litter
    leftovers = [p.name for p in staged_root.parent.iterdir() if p.name != record["sha256"]]
    assert leftovers == []


def test_second_stage_reuses_warm_copy(install_root: Path, tmp_path: Path, capsys):
    _pack_and_record(install_root)
    base = tmp_path / "local"
    base.mkdir()

    first = stage_env(install_root, ENV_NAME, base)
    second = stage_env(install_root, ENV_NAME, base)

    assert first == second
    assert "Stage reused (warm)" in capsys.readouterr().err


def test_rebuilt_env_stages_beside_old_copy(install_root: Path, tmp_path: Path):
    _pack_and_record(install_root)
    base = tmp_path / "local"
    base.mkdir()
    first = stage_env(install_root, ENV_NAME, base)

    # Rebuild: env content changes, repack, manifest re-records.
    site = install_root / "envs" / ENV_NAME / "lib" / "python3.11" / "site-packages"
    (site / "v2.so").write_bytes(b"z" * 4096)
    _pack_and_record(
        install_root, built_at="2026-09-02T00:00:00Z", packed_at="2026-09-02T00:00:01Z"
    )
    second = stage_env(install_root, ENV_NAME, base)

    assert first is not None and second is not None
    assert first != second  # content-addressed: new build, new dir
    assert (second / "envs" / ENV_NAME / "lib" / "python3.11" / "site-packages" / "v2.so").exists()


def test_stale_image_falls_back(install_root: Path, tmp_path: Path):
    # Env rebuilt after the pack: built_at is newer than packed_at.
    _pack_and_record(
        install_root, built_at="2026-09-03T00:00:00Z", packed_at="2026-09-01T00:00:01Z"
    )
    base = tmp_path / "local"
    base.mkdir()
    assert stage_env(install_root, ENV_NAME, base) is None


def test_missing_archive_falls_back(install_root: Path, tmp_path: Path):
    record = _pack_and_record(install_root)
    (install_root / record["path"]).unlink()
    base = tmp_path / "local"
    base.mkdir()
    assert stage_env(install_root, ENV_NAME, base) is None


def test_no_image_record_falls_back(install_root: Path, tmp_path: Path):
    write_manifest_env(install_root, image=None)
    base = tmp_path / "local"
    base.mkdir()
    assert stage_env(install_root, ENV_NAME, base) is None


def test_corrupt_archive_falls_back_and_cleans_up(install_root: Path, tmp_path: Path, capsys):
    record = _pack_and_record(install_root)
    (install_root / record["path"]).write_bytes(b"not a zstd stream")
    base = tmp_path / "local"
    base.mkdir()

    assert stage_env(install_root, ENV_NAME, base) is None
    assert "falling back to prewarm" in capsys.readouterr().err
    import getpass

    envs_root = base / "rootstock" / getpass.getuser() / "envs-by-hash"
    assert not any(envs_root.glob("*.partial.*"))
    assert not any(envs_root.glob("*.lock"))


def test_insufficient_space_falls_back(install_root: Path, tmp_path: Path, monkeypatch, capsys):
    _pack_and_record(install_root)
    base = tmp_path / "local"
    base.mkdir()

    import shutil as _shutil

    usage = _shutil.disk_usage(base)
    monkeypatch.setattr("rootstock.stage.shutil.disk_usage", lambda p: usage._replace(free=10))
    assert stage_env(install_root, ENV_NAME, base) is None
    assert "free at" in capsys.readouterr().err
