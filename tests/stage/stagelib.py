"""Shared helpers for the pack/stage tests: a miniature install root with
one built env whose venv symlinks and pyvenv.cfg point at the root's
.python/, the way uv-built envs do on a real install."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

requires_archive_tools = pytest.mark.skipif(
    shutil.which("tar") is None or shutil.which("zstd") is None,
    reason="tar/zstd not on PATH",
)

ENV_NAME = "demo"
INTERP = "cpython-3.11.99-test"


def build_install_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    interp_bin = root / ".python" / INTERP / "bin"
    interp_bin.mkdir(parents=True)
    (interp_bin / "python3.11").write_text("#!/bin/sh\n")
    (interp_bin / "python3.11").chmod(0o755)

    env_bin = root / "envs" / ENV_NAME / "bin"
    env_bin.mkdir(parents=True)
    os.symlink(interp_bin / "python3.11", env_bin / "python")
    os.symlink("python", env_bin / "python3")

    env_dir = root / "envs" / ENV_NAME
    (env_dir / "pyvenv.cfg").write_text(
        f"home = {interp_bin}\nversion_info = 3.11.99\nrelocatable = true\n"
    )
    site = env_dir / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    (site / "libdemo.so").write_bytes(b"\x7fELF" + b"x" * 1000)
    (env_dir / "env_source.py").write_text("CHECKPOINTS = {}\n")
    return root


def write_manifest_env(
    root: Path,
    env_name: str = ENV_NAME,
    built_at: str = "2026-09-01T00:00:00Z",
    image: dict | None = None,
    checkpoints: dict | None = None,
) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 7,
                "environments": {
                    env_name: {
                        "built_at": built_at,
                        "image": image,
                        "checkpoints": checkpoints or {},
                    }
                },
            }
        )
    )
