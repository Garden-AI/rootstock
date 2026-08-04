"""built_at semantics: stamped by install, preserved on refresh, estimated
from disk for envs the manifest has never seen.

`verified_at > built_at` is the staleness comparison behind `rootstock
status` and smoke-test reporting; a fabricated built_at silently breaks it
in both directions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rootstock.manifest import (
    SCHEMA_VERSION,
    Maintainer,
    Manifest,
    built_at_estimate,
)
from rootstock.operations import _ensure_manifest_entry, refresh_manifest_environments

ENV_SOURCE = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    '# dependencies = ["six>=1.0"]\n'
    "# ///\n"
    "CHECKPOINTS = {}\n"
)

BUILD_TIME = 1735689600  # 2025-01-01T00:00:00Z


@pytest.fixture(autouse=True)
def _no_version_probe(monkeypatch):
    """Version probing shells out to the env's python; not under test here."""
    monkeypatch.setattr("rootstock.operations.get_installed_versions", lambda *a, **k: {})


def _make_built_env(root: Path, name: str = "mace") -> Path:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(ENV_SOURCE)
    os.utime(env_dir, (BUILD_TIME, BUILD_TIME))
    return env_dir


def _manifest(root: Path, environments=None) -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        clusters=["test"],
        root=str(root),
        maintainer=Maintainer(name="a", email="a@b.c"),
        rootstock_version="0.0.0",
        python_version="3.10",
        last_updated="2026-01-01T00:00:00Z",
        environments=environments or {},
    )


# --- _refresh_manifest_environments ------------------------------------------


def test_refresh_stamps_built_env_to_now(tmp_path):
    _make_built_env(tmp_path)
    manifest = refresh_manifest_environments(_manifest(tmp_path), tmp_path)
    old = manifest.environments["mace"].built_at

    manifest = refresh_manifest_environments(manifest, tmp_path, built_env="mace")

    assert manifest.environments["mace"].built_at > old


def test_refresh_preserves_built_at_for_known_env(tmp_path):
    _make_built_env(tmp_path)
    manifest = refresh_manifest_environments(_manifest(tmp_path), tmp_path)
    recorded = manifest.environments["mace"].built_at

    manifest = refresh_manifest_environments(manifest, tmp_path)

    assert manifest.environments["mace"].built_at == recorded


def test_refresh_estimates_built_at_from_dir_mtime_for_unknown_env(tmp_path):
    env_dir = _make_built_env(tmp_path)

    manifest = refresh_manifest_environments(_manifest(tmp_path), tmp_path)

    built_at = manifest.environments["mace"].built_at
    assert built_at == built_at_estimate(env_dir)
    assert built_at.startswith("2025-01-01")  # dir mtime, not now


# --- rootstock add: no more placeholder entries -------------------------------


def test_add_backfills_real_env_info_from_disk(tmp_path):
    """When the manifest lags a built env, add refreshes from disk instead of
    synthesizing a placeholder with source_hash='' and built_at=now."""
    env_dir = _make_built_env(tmp_path)

    env, _ = _ensure_manifest_entry(_manifest(tmp_path), tmp_path, "mace", "some-ckpt")

    assert env.source_hash.startswith("sha256:")
    assert env.source == ENV_SOURCE
    assert env.built_at == built_at_estimate(env_dir)
    assert "some-ckpt" in env.checkpoints


def test_add_errors_on_unbuilt_env(tmp_path):
    with pytest.raises(RuntimeError, match="not built"):
        _ensure_manifest_entry(_manifest(tmp_path), tmp_path, "mace", "some-ckpt")


def test_add_errors_on_env_missing_source(tmp_path):
    env_dir = tmp_path / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()

    with pytest.raises(RuntimeError, match="no env_source.py"):
        _ensure_manifest_entry(_manifest(tmp_path), tmp_path, "mace", "some-ckpt")
