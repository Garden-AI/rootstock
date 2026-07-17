"""_refresh_manifest_environments records the built env's lockfile hash."""

from __future__ import annotations

from pathlib import Path

from rootstock.manifest import SCHEMA_VERSION, Maintainer, Manifest, compute_source_hash
from rootstock.operations import refresh_manifest_environments

ENV_SOURCE = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    '# dependencies = ["six>=1.0"]\n'
    "# ///\n"
    "CHECKPOINTS = {}\n"
)


def _make_built_env(root: Path, name: str, with_lock: bool) -> Path:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(ENV_SOURCE)
    if with_lock:
        (env_dir / "env_source.py.lock").write_text("version = 1\n")
    return env_dir


def _empty_manifest(root: Path) -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        cluster="test",
        root=str(root),
        maintainer=Maintainer(name="a", email="a@b.c"),
        rootstock_version="0.0.0",
        python_version="3.10",
        last_updated="2026-01-01T00:00:00Z",
    )


def test_refresh_records_lock_hash(tmp_path: Path, monkeypatch):
    # Version probing shells out to the env's python; not under test here.
    monkeypatch.setattr("rootstock.operations.get_installed_versions", lambda *a, **k: {})
    env_dir = _make_built_env(tmp_path, "locked", with_lock=True)

    manifest = refresh_manifest_environments(_empty_manifest(tmp_path), tmp_path)

    expected = compute_source_hash(env_dir / "env_source.py.lock")
    assert manifest.environments["locked"].lock_hash == expected


def test_refresh_without_lockfile_records_none(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rootstock.operations.get_installed_versions", lambda *a, **k: {})
    _make_built_env(tmp_path, "legacy", with_lock=False)

    manifest = refresh_manifest_environments(_empty_manifest(tmp_path), tmp_path)

    assert manifest.environments["legacy"].lock_hash is None
