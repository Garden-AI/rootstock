"""
Tests for the v1 -> v2 manifest migration script.

# TODO(rootstock-v0.8.0+1): delete this file together with
# scripts/migrate_manifest_v1_to_v2.py once known clusters are migrated.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rootstock.manifest import Manifest

MIGRATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrate_manifest_v1_to_v2.py"


def _load_migrate_module():
    spec = importlib.util.spec_from_file_location("migrate_v1_v2", MIGRATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migrate():
    return _load_migrate_module().migrate


def _v1_manifest() -> dict:
    return {
        "schema_version": "1",
        "cluster": "modal",
        "root": "/vol/rootstock",
        "maintainer": {"name": "Will", "email": "w@e.com"},
        "rootstock_version": "0.7.3",
        "python_version": "3.10.19",
        "last_updated": "2026-04-01T00:00:00Z",
        "environments": {
            "mace_env": {
                "status": "ready",
                "built_at": "2026-04-01T00:00:00Z",
                "source_hash": "sha256:abc",
                "source": "",
                "python_requires": ">=3.10",
                "dependencies": {"mace-torch": "0.3.6"},
                "checkpoints": ["small", "medium", "large"],
            },
            "uma_env": {
                "status": "ready",
                "built_at": "2026-04-01T00:00:00Z",
                "source_hash": "sha256:def",
                "source": "",
                "python_requires": ">=3.10",
                "dependencies": {},
                "checkpoints": [],
            },
        },
    }


def test_migrate_converts_checkpoint_list_to_dict(migrate):
    out = migrate(_v1_manifest())
    assert out["schema_version"] == 2
    mace_ckpts = out["environments"]["mace_env"]["checkpoints"]
    assert set(mace_ckpts.keys()) == {"small", "medium", "large"}
    for ckpt in mace_ckpts.values():
        assert ckpt == {
            "fetched_at": None,
            "verified_at": None,
            "verified_device": None,
            "last_error": None,
        }


def test_migrate_handles_empty_checkpoints(migrate):
    out = migrate(_v1_manifest())
    assert out["environments"]["uma_env"]["checkpoints"] == {}


def test_migrate_output_loads_as_v2(migrate):
    out = migrate(_v1_manifest())
    m = Manifest.from_dict(out)
    assert m.schema_version == 2
    assert set(m.environments["mace_env"].checkpoints.keys()) == {"small", "medium", "large"}


def test_migrate_idempotent_on_v2(migrate):
    once = migrate(_v1_manifest())
    twice = migrate(once)
    assert once == twice


def test_migrate_rejects_unexpected_version(migrate):
    bad = _v1_manifest()
    bad["schema_version"] = 99
    with pytest.raises(SystemExit):
        migrate(bad)


def test_migrate_cli_writes_backup_and_rewrites_in_place(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_v1_manifest()))

    module = _load_migrate_module()
    rc = module.main([str(MIGRATE_PATH), str(manifest_path)])
    assert rc == 0

    backup = manifest_path.with_suffix(".json.v1.bak")
    assert backup.exists()
    assert json.loads(backup.read_text())["schema_version"] == "1"

    new_data = json.loads(manifest_path.read_text())
    assert new_data["schema_version"] == 2
    Manifest.from_dict(new_data)  # round-trips
