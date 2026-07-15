"""Manifest schema migrations.

Old manifests persist on every deployed cluster root, and "reinstall all
environments and re-add all checkpoints" is a week of cluster work under the
pre-1.0 cost model — so a schema bump must come with migration code, and
loading any historical manifest must succeed.
"""

from __future__ import annotations

import copy
import json

import pytest

import rootstock.manifest as manifest_module
from rootstock.manifest import (
    SCHEMA_VERSION,
    Manifest,
    load_manifest,
    migrate_manifest_data,
)


def _base(version, environments=None) -> dict:
    return {
        "schema_version": version,
        "cluster": "test",
        "root": "/tmp/x",
        "maintainer": {"name": "a", "email": "a@b.c"},
        "rootstock_version": "0.5.0",
        "python_version": "3.10",
        "last_updated": "2026-01-01T00:00:00Z",
        "environments": environments or {},
    }


def _v2_env(checkpoints=None) -> dict:
    return {
        "status": "ready",
        "built_at": "2026-01-01T00:00:00Z",
        "source_hash": "sha256:abc",
        "source": "CHECKPOINTS = {}",
        "python_requires": ">=3.10",
        "dependencies": {"mace-torch": "0.3.6"},
        "checkpoints": checkpoints or {},
    }


def _v1_env(checkpoints: list[str]) -> dict:
    env = _v2_env()
    env["checkpoints"] = checkpoints  # v1: a bare list of names
    return env


# --- v2 -> v3 -------------------------------------------------------------


def test_v2_environments_survive_checkpoints_dropped():
    ckpt = {"medium": {"fetched_at": "2026-01-02T00:00:00Z"}}
    data = _base(2, {"mace": _v2_env(ckpt)})

    migrated, notes = migrate_manifest_data(data)

    assert migrated["schema_version"] == SCHEMA_VERSION
    env = migrated["environments"]["mace"]
    assert env["dependencies"] == {"mace-torch": "0.3.6"}
    # v2 checkpoint keys aren't canonical ids; they can't be trusted to join
    assert env["checkpoints"] == {}
    assert any("rootstock add" in n for n in notes)


def test_v2_without_checkpoints_migrates_quietly():
    _, notes = migrate_manifest_data(_base(2, {"mace": _v2_env()}))
    assert notes == ["migrated manifest schema v2 -> v3"]


def test_v2_loads_via_from_dict():
    manifest = Manifest.from_dict(_base(2, {"mace": _v2_env()}))
    assert manifest.schema_version == SCHEMA_VERSION
    assert "mace" in manifest.environments


# --- v1 -> v3 (full chain) --------------------------------------------------


def test_v1_chain_migrates_to_current():
    data = _base("1", {"mace": _v1_env(["small", "medium"])})

    migrated, notes = migrate_manifest_data(data)

    assert migrated["schema_version"] == SCHEMA_VERSION
    # v1->v2 mints empty CheckpointInfo dicts; v2->v3 then drops them
    assert migrated["environments"]["mace"]["checkpoints"] == {}
    assert len(notes) == 2
    assert Manifest.from_dict(migrated).environments["mace"].source_hash == "sha256:abc"


# --- guard rails ------------------------------------------------------------


def test_current_version_passes_through_unchanged():
    data = _base(SCHEMA_VERSION)
    migrated, notes = migrate_manifest_data(data)
    assert migrated is data
    assert notes == []


def test_caller_dict_is_not_mutated():
    data = _base(2, {"mace": _v2_env({"medium": {"fetched_at": "x"}})})
    snapshot = copy.deepcopy(data)

    migrate_manifest_data(data)

    assert data == snapshot


def test_newer_manifest_tells_user_to_upgrade_rootstock():
    with pytest.raises(RuntimeError, match="upgrade this client"):
        migrate_manifest_data(_base(SCHEMA_VERSION + 1))


def test_missing_version_rejected():
    data = _base(3)
    del data["schema_version"]
    with pytest.raises(RuntimeError, match="invalid schema_version"):
        migrate_manifest_data(data)


def test_schema_bump_without_migration_is_loud(monkeypatch):
    """If SCHEMA_VERSION is ever bumped without a migration, every deployed
    manifest of the previous version must fail with a clear error — not
    silently misparse."""
    monkeypatch.setattr(manifest_module, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    with pytest.raises(RuntimeError, match="no migration path"):
        migrate_manifest_data(_base(SCHEMA_VERSION))


# --- load_manifest boundary --------------------------------------------------


def test_load_manifest_migrates_and_notes(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(
        json.dumps(_base(2, {"mace": _v2_env({"medium": {"fetched_at": "x"}})}))
    )

    manifest = load_manifest(tmp_path)

    assert manifest is not None
    assert manifest.schema_version == SCHEMA_VERSION
    err = capsys.readouterr().err
    assert "migrated manifest schema v2 -> v3" in err
    assert "written back on the next state-changing command" in err


def test_load_manifest_current_version_prints_nothing(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(json.dumps(_base(SCHEMA_VERSION)))

    assert load_manifest(tmp_path) is not None
    assert capsys.readouterr().err == ""
