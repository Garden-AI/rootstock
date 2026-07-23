"""Tests for the per-user local-checkpoint registry."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from rootstock.local_checkpoints import (
    LocalCheckpointEntry,
    LocalCheckpointError,
    hash_weights_file,
    load_local_registry,
    local_checkpoints_for_root,
    record_local_verification,
    register_local_checkpoint,
    remove_local_checkpoint,
    save_local_registry,
)

_ENV_WITH_HOOK = """\
CHECKPOINTS = {"uma-s-1p1": "uma-s-1p1"}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""

_ENV_WITHOUT_HOOK = """\
CHECKPOINTS = {"mace-mp-0-medium": "medium"}


def setup(checkpoint, device="cuda"):
    return None
"""


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """Install root with a hook-declaring 'uma' env and a plain 'mace' env."""
    root = tmp_path / "root"
    for name, source in (("uma", _ENV_WITH_HOOK), ("mace", _ENV_WITHOUT_HOOK)):
        env_dir = root / "envs" / name
        (env_dir / "bin").mkdir(parents=True)
        (env_dir / "bin" / "python").touch()
        (env_dir / "env_source.py").write_text(source)
    return root


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "my-uma-ft.pt"
    path.write_bytes(b"weights bytes " * 100)
    return path


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    return tmp_path / "registry.json"


# ---------- hash_weights_file ----------------------------------------------


def test_hash_weights_file(weights):
    sha, size = hash_weights_file(weights)
    expected = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert sha == f"sha256:{expected}"
    assert size == weights.stat().st_size


# ---------- load / save round-trip ------------------------------------------


def test_load_missing_file_is_empty(registry):
    assert load_local_registry(registry) == {}


def test_round_trip(registry):
    entry = LocalCheckpointEntry(
        env="uma",
        path="/scratch/me/ft.pt",
        sha256="sha256:abc",
        size=42,
        setup_kwargs={"task": "omol"},
        registered_at="2026-07-21T00:00:00+00:00",
    )
    save_local_registry({"/some/root": {"my-ft": entry}}, registry)
    loaded = load_local_registry(registry)
    assert loaded["/some/root"]["my-ft"] == entry


def test_save_leaves_no_temp_files(registry):
    save_local_registry({}, registry)
    siblings = [p.name for p in registry.parent.iterdir()]
    assert siblings == [registry.name]


def test_corrupt_registry_raises(registry):
    registry.write_text("{not json")
    with pytest.raises(LocalCheckpointError, match="corrupted"):
        load_local_registry(registry)


def test_newer_schema_raises(registry):
    registry.write_text(json.dumps({"schema_version": 99, "roots": {}}))
    with pytest.raises(LocalCheckpointError, match="newer rootstock"):
        load_local_registry(registry)


# ---------- register validations --------------------------------------------


def test_register_happy_path(fake_root, weights, registry):
    entry = register_local_checkpoint(
        fake_root,
        "my-uma-ft",
        "uma",
        weights,
        setup_kwargs={"task": "omol"},
        registry_path=registry,
    )
    assert entry.env == "uma"
    assert entry.path == str(weights.resolve())
    assert entry.sha256.startswith("sha256:")
    assert entry.size == weights.stat().st_size
    assert entry.verified_at is None

    on_disk = local_checkpoints_for_root(fake_root, registry_path=registry)
    assert on_disk["my-uma-ft"] == entry


def test_register_missing_weights_file(fake_root, registry, tmp_path):
    with pytest.raises(LocalCheckpointError, match="not found"):
        register_local_checkpoint(
            fake_root, "my-ft", "uma", tmp_path / "nope.pt", registry_path=registry
        )


def test_register_unbuilt_env(fake_root, weights, registry):
    with pytest.raises(LocalCheckpointError, match="not built"):
        register_local_checkpoint(fake_root, "my-ft", "tensornet", weights, registry_path=registry)


def test_register_env_without_hook(fake_root, weights, registry):
    with pytest.raises(LocalCheckpointError, match="setup_from_path") as exc:
        register_local_checkpoint(fake_root, "my-ft", "mace", weights, registry_path=registry)
    # The error points at envs that DO support local checkpoints.
    assert "uma" in str(exc.value)


def test_register_canonical_collision(fake_root, weights, registry):
    with pytest.raises(LocalCheckpointError, match="canonical"):
        register_local_checkpoint(fake_root, "uma-s-1p1", "uma", weights, registry_path=registry)


@pytest.mark.parametrize("key", ["checkpoint", "device", "path"])
def test_register_reserved_kwargs(fake_root, weights, registry, key):
    with pytest.raises(LocalCheckpointError, match="reserved"):
        register_local_checkpoint(
            fake_root,
            "my-ft",
            "uma",
            weights,
            setup_kwargs={key: "x"},
            registry_path=registry,
        )


def test_reregister_overwrites_and_resets_verification(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    record_local_verification(fake_root, "my-ft", ok=True, device="cuda", registry_path=registry)
    assert (
        local_checkpoints_for_root(fake_root, registry_path=registry)["my-ft"].verified_at
        is not None
    )

    entry = register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    assert entry.verified_at is None
    assert entry.last_error is None


def test_root_key_normalization(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    # Reaching the same root through ".." resolves to the same entry set.
    indirect = fake_root / "envs" / ".."
    assert "my-ft" in local_checkpoints_for_root(indirect, registry_path=registry)


# ---------- remove -----------------------------------------------------------


def test_remove_happy_path(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    entry = remove_local_checkpoint(fake_root, "my-ft", registry_path=registry)
    assert entry.path == str(weights.resolve())
    assert local_checkpoints_for_root(fake_root, registry_path=registry) == {}
    # Weights file untouched.
    assert weights.exists()
    # Empty root key pruned from the file.
    assert load_local_registry(registry) == {}


def test_remove_unknown_id_lists_registered(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    with pytest.raises(LocalCheckpointError, match="my-ft"):
        remove_local_checkpoint(fake_root, "typo", registry_path=registry)


def test_remove_from_empty_registry(fake_root, registry):
    with pytest.raises(LocalCheckpointError, match="No local checkpoints"):
        remove_local_checkpoint(fake_root, "my-ft", registry_path=registry)


# ---------- record_local_verification ----------------------------------------


def test_record_verification_success(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    record_local_verification(fake_root, "my-ft", ok=True, device="cuda", registry_path=registry)
    entry = local_checkpoints_for_root(fake_root, registry_path=registry)["my-ft"]
    assert entry.verified_at is not None
    assert entry.verified_device == "cuda"
    assert entry.last_error is None


def test_record_verification_failure(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    record_local_verification(
        fake_root,
        "my-ft",
        ok=False,
        device="cuda",
        error="verify: boom",
        registry_path=registry,
    )
    entry = local_checkpoints_for_root(fake_root, registry_path=registry)["my-ft"]
    assert entry.verified_at is None
    assert entry.last_error == "verify: boom"


def test_record_verification_failure_revokes_earlier_success(fake_root, weights, registry):
    # Same semantics as the manifest: a failure clears verified state, so a
    # previously-verified checkpoint can't show ✓ alongside last_error.
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    record_local_verification(fake_root, "my-ft", ok=True, device="cuda", registry_path=registry)
    record_local_verification(
        fake_root,
        "my-ft",
        ok=False,
        device="cuda",
        error="smoke-test: boom",
        registry_path=registry,
    )
    entry = local_checkpoints_for_root(fake_root, registry_path=registry)["my-ft"]
    assert entry.verified_at is None
    assert entry.verified_device is None
    assert entry.last_error == "smoke-test: boom"


def test_record_verification_for_removed_id_is_noop(fake_root, registry):
    # Outcome for an id removed meanwhile has nothing to attach to.
    record_local_verification(fake_root, "gone", ok=True, device="cuda", registry_path=registry)
    assert load_local_registry(registry) == {}


# ---------- permissions -------------------------------------------------------


def test_registry_file_is_private(fake_root, weights, registry):
    register_local_checkpoint(fake_root, "my-ft", "uma", weights, registry_path=registry)
    mode = os.stat(registry).st_mode & 0o777
    assert mode == 0o600
