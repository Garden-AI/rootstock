"""_refresh_manifest_environments drops records for envs gone from disk.

The filesystem is the truth for what's installed; a manifest record with no
built env behind it must not survive a refresh (and hence must never reach
the push payload the Almanac renders from).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.manifest import (
    SCHEMA_VERSION,
    EnvironmentInfo,
    Maintainer,
    Manifest,
)
from rootstock.operations import refresh_manifest_environments

ENV_SOURCE = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    '# dependencies = ["six>=1.0"]\n'
    "# ///\n"
    "CHECKPOINTS = {}\n"
)


@pytest.fixture(autouse=True)
def _no_version_probe(monkeypatch):
    """Version probing shells out to the env's python; not under test here."""
    monkeypatch.setattr("rootstock.operations.get_installed_versions", lambda *a, **k: {})


def _make_built_env(root: Path, name: str, with_source: bool = True) -> Path:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    if with_source:
        (env_dir / "env_source.py").write_text(ENV_SOURCE)
    return env_dir


def _record() -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
    )


def _manifest(root: Path, environments: dict) -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        clusters=["test"],
        root=str(root),
        maintainer=Maintainer(name="a", email="a@b.c"),
        rootstock_version="0.0.0",
        python_version="3.10",
        last_updated="2026-01-01T00:00:00Z",
        environments=environments,
    )


def test_refresh_drops_record_for_env_gone_from_disk(tmp_path, capsys):
    _make_built_env(tmp_path, "mace")
    manifest = _manifest(tmp_path, {"mace": _record(), "deleted": _record()})

    manifest = refresh_manifest_environments(manifest, tmp_path)

    assert set(manifest.environments) == {"mace"}
    assert "dropping manifest record for 'deleted'" in capsys.readouterr().err


def test_refresh_keeps_record_for_built_env_missing_source(tmp_path, capsys):
    """A built env whose env_source.py is missing is still installed; its
    existing record is kept untouched rather than pruned or refreshed."""
    _make_built_env(tmp_path, "sourceless", with_source=False)
    record = _record()
    manifest = _manifest(tmp_path, {"sourceless": record})

    manifest = refresh_manifest_environments(manifest, tmp_path)

    assert manifest.environments["sourceless"] is record
    assert "dropping" not in capsys.readouterr().err
