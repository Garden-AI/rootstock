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
