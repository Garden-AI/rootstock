"""Tests for the v3 manifest schema."""

from __future__ import annotations

import pytest

from rootstock.manifest import (
    SCHEMA_VERSION,
    CheckpointInfo,
    EnvironmentInfo,
    Maintainer,
    Manifest,
    is_verified,
)


def _make_env(built_at: str = "2026-01-01T00:00:00Z", **ckpts: CheckpointInfo) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at=built_at,
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints=dict(ckpts),
    )


def _make_manifest(envs: dict[str, EnvironmentInfo] | None = None) -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        cluster="test",
        root="/tmp/x",
        maintainer=Maintainer(name="a", email="a@b.c"),
        rootstock_version="0.0.0",
        python_version="3.10",
        last_updated="2026-01-01T00:00:00Z",
        environments=envs or {},
    )


def test_schema_version_constant():
    assert SCHEMA_VERSION == 4


def test_checkpoint_info_round_trip():
    ckpt = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z",
        verified_at="2026-01-03T00:00:00Z",
        verified_device="cuda",
        last_error=None,
    )
    assert CheckpointInfo.from_dict(ckpt.to_dict()) == ckpt


def test_checkpoint_info_defaults_to_none():
    ckpt = CheckpointInfo()
    assert ckpt.fetched_at is None
    assert ckpt.verified_at is None
    assert ckpt.verified_device is None
    assert ckpt.last_error is None


def test_environment_info_with_dict_checkpoints_round_trip():
    env = _make_env(
        **{
            "mace-mp-0-medium": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
        }
    )
    restored = EnvironmentInfo.from_dict(env.to_dict())
    assert "mace-mp-0-medium" in restored.checkpoints
    assert restored.checkpoints["mace-mp-0-medium"].fetched_at == "2026-01-02T00:00:00Z"


def test_environment_info_lock_hash_round_trip():
    env = _make_env()
    env.lock_hash = "sha256:def456"
    restored = EnvironmentInfo.from_dict(env.to_dict())
    assert restored.lock_hash == "sha256:def456"


def test_environment_info_without_lock_hash_loads_as_none():
    """v3 manifests written before lockfiles existed have no lock_hash key."""
    data = _make_env().to_dict()
    del data["lock_hash"]
    assert EnvironmentInfo.from_dict(data).lock_hash is None


def test_manifest_round_trip_preserves_checkpoint_metadata():
    m = _make_manifest(
        {
            "mace": _make_env(
                **{
                    "mace-mp-0-medium": CheckpointInfo(
                        fetched_at="2026-01-02T00:00:00Z",
                        verified_at="2026-01-03T00:00:00Z",
                        verified_device="cuda",
                    ),
                    "mace-mp-0-small": CheckpointInfo(),
                }
            )
        }
    )
    restored = Manifest.from_dict(m.to_dict())
    ckpts = restored.environments["mace"].checkpoints
    assert ckpts["mace-mp-0-medium"].verified_device == "cuda"
    assert ckpts["mace-mp-0-small"].fetched_at is None


def test_from_dict_migrates_v2():
    """Old manifests load via migration instead of demanding a reinstall.

    (Migration specifics are covered in test_migrations.py.)
    """
    v2 = {
        "schema_version": 2,
        "cluster": "x",
        "root": "/",
        "maintainer": {"name": "a", "email": "b"},
        "rootstock_version": "0",
        "python_version": "0",
        "last_updated": "0",
    }
    manifest = Manifest.from_dict(v2)
    assert manifest.schema_version == SCHEMA_VERSION


def test_from_dict_coerces_digit_string_version():
    """v1-era writers stored schema_version as a string; tolerate that."""
    stringly = {
        "schema_version": "3",
        "cluster": "x",
        "root": "/",
        "maintainer": {"name": "a", "email": "b"},
        "rootstock_version": "0",
        "python_version": "0",
        "last_updated": "0",
    }
    assert Manifest.from_dict(stringly).schema_version == SCHEMA_VERSION


def test_from_dict_rejects_missing_schema_version():
    no_version = {
        "cluster": "x",
        "root": "/",
        "maintainer": {"name": "a", "email": "b"},
        "rootstock_version": "0",
        "python_version": "0",
        "last_updated": "0",
    }
    with pytest.raises(RuntimeError, match="schema_version"):
        Manifest.from_dict(no_version)


def test_is_verified_unverified_checkpoint():
    env = _make_env()
    ckpt = CheckpointInfo(fetched_at="2026-01-02T00:00:00Z")
    assert is_verified(env, ckpt) is False


def test_is_verified_after_build():
    env = _make_env(built_at="2026-01-01T00:00:00Z")
    ckpt = CheckpointInfo(verified_at="2026-01-02T00:00:00Z", verified_device="cuda")
    assert is_verified(env, ckpt) is True


def test_is_verified_stale_when_env_rebuilt_after_verify():
    env = _make_env(built_at="2026-02-01T00:00:00Z")
    ckpt = CheckpointInfo(verified_at="2026-01-15T00:00:00Z", verified_device="cuda")
    assert is_verified(env, ckpt) is False
