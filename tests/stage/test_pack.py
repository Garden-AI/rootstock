"""Packing envs into single-image archives (#180)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from stagelib import ENV_NAME, INTERP, requires_archive_tools

from rootstock.pack import (
    PackError,
    env_interpreter_dir,
    pack_environment,
    pack_environment_best_effort,
)


def test_interpreter_dir_resolves_through_venv_symlink(install_root: Path):
    assert env_interpreter_dir(install_root, ENV_NAME) == install_root / ".python" / INTERP


def test_interpreter_outside_root_refuses(install_root: Path, tmp_path: Path):
    foreign = tmp_path / "foreign-python"
    foreign.write_text("#!/bin/sh\n")
    python_link = install_root / "envs" / ENV_NAME / "bin" / "python"
    python_link.unlink()
    python_link.symlink_to(foreign)
    with pytest.raises(PackError, match="outside"):
        env_interpreter_dir(install_root, ENV_NAME)


def test_pack_unbuilt_env_refuses(install_root: Path):
    with pytest.raises(PackError, match="not built"):
        pack_environment(install_root, "nope")


@requires_archive_tools
def test_pack_produces_verifiable_archive(install_root: Path):
    record = pack_environment(install_root, ENV_NAME)

    image = install_root / record["path"]
    assert image.is_file()
    assert record["format"] == "tar.zst"
    assert record["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert image.name == f"{ENV_NAME}-{record['sha256'][:12]}.tar.zst"
    assert record["compressed_bytes"] == image.stat().st_size
    assert record["uncompressed_bytes"] > 0
    # "packed_at" is stamped by the manifest refresh, not the packer.
    assert "packed_at" not in record

    # The archive holds root-relative paths for the env AND its interpreter.
    listing = subprocess.run(
        f"zstd -dc {image} | tar -tf -", shell=True, capture_output=True, text=True, check=True
    ).stdout
    assert f"envs/{ENV_NAME}/pyvenv.cfg" in listing
    assert f".python/{INTERP}/bin/python3.11" in listing


@requires_archive_tools
def test_repack_removes_superseded_images(install_root: Path):
    first = pack_environment(install_root, ENV_NAME)
    # Change the env so the archive bytes (and sha) differ.
    site = install_root / "envs" / ENV_NAME / "lib" / "python3.11" / "site-packages"
    (site / "extra.so").write_bytes(b"y" * 2048)
    second = pack_environment(install_root, ENV_NAME)

    assert first["sha256"] != second["sha256"]
    images = sorted(p.name for p in (install_root / "images").iterdir())
    assert images == [Path(second["path"]).name]


def test_best_effort_pack_degrades_to_warning(install_root: Path, capsys):
    assert pack_environment_best_effort(install_root, "nope") is None
    assert "rootstock pack" in capsys.readouterr().err


@requires_archive_tools
def test_pack_spares_live_concurrent_partials(install_root: Path):
    """install's auto-pack racing a batch `rootstock pack` of the same env:
    the winner must not sweep the loser's in-flight partial."""
    import subprocess as sp

    images = install_root / "images"
    images.mkdir()
    live = images / ".demo.packing.1"  # pid 1 is always alive
    live.write_bytes(b"x")
    reaped = sp.Popen(["true"])
    reaped.wait()
    dead = images / f".demo.packing.{reaped.pid}"
    dead.write_bytes(b"x")

    pack_environment(install_root, ENV_NAME)

    assert live.exists()
    assert not dead.exists()


@requires_archive_tools
def test_repack_spares_dash_extended_sibling_env_images(install_root: Path):
    """Cleanup matches `<env>-<12 hex>.tar.zst` exactly: packing 'demo' must
    never delete 'demo-tuned-<sha>.tar.zst'."""
    images = install_root / "images"
    images.mkdir()
    sibling = images / "demo-tuned-0123456789ab.tar.zst"
    sibling.write_bytes(b"z")
    old_own = images / "demo-ba9876543210.tar.zst"
    old_own.write_bytes(b"z")

    pack_environment(install_root, ENV_NAME)

    assert sibling.exists()
    assert not old_own.exists()


def test_image_usable_requires_the_archive_on_disk(tmp_path: Path):
    """The pack sweep's filter must see through a purged images/ dir — a
    current-looking record with no file behind it means repack, not 'all
    current'."""
    from rootstock.manifest import EnvironmentInfo
    from rootstock.operations import _image_usable

    image = {"path": "images/demo-abc123def456.tar.zst", "packed_at": "2026-09-01T00:00:01Z"}
    env = EnvironmentInfo(
        built_at="2026-09-01T00:00:00Z",
        source_hash=None,
        source="",
        python_requires=">=3.11",
        dependencies={},
        image=image,
    )
    assert not _image_usable(tmp_path, env)  # record fine, file purged
    target = tmp_path / image["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"z")
    assert _image_usable(tmp_path, env)
