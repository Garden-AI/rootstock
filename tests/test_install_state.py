"""InstallState: filesystem decides what's installed; manifest supplies the rest."""

from __future__ import annotations

from pathlib import Path

from rootstock.install_state import read_install_state
from rootstock.manifest import (
    SCHEMA_VERSION,
    EnvironmentInfo,
    Maintainer,
    Manifest,
    compute_source_hash,
    save_manifest,
)

ENV_SOURCE = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    '# dependencies = ["six>=1.0"]\n'
    "# ///\n"
    'CHECKPOINTS = {"mace-mp-0-small": "small"}\n'
)


def _make_built_env(root: Path, name: str = "mace", source: str | None = ENV_SOURCE) -> Path:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    if source is not None:
        (env_dir / "env_source.py").write_text(source)
    return env_dir


def _env_record(source_hash: str = "sha256:abc") -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at="2026-01-01T00:00:00Z",
        source_hash=source_hash,
        source="",
        python_requires=">=3.10",
        dependencies={},
    )


def _manifest(root: Path, environments: dict | None = None) -> Manifest:
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


def test_envs_enumerated_from_filesystem_without_manifest(tmp_path):
    _make_built_env(tmp_path)

    state = read_install_state(tmp_path)

    assert state.manifest is None
    assert set(state.envs) == {"mace"}
    env = state.envs["mace"]
    assert env.source_hash == compute_source_hash(env.source_file)
    assert env.declared_checkpoints == {"mace-mp-0-small": "small"}
    assert env.record is None


def test_manifest_record_joined_when_present(tmp_path):
    env_dir = _make_built_env(tmp_path)
    source_hash = compute_source_hash(env_dir / "env_source.py")
    manifest = _manifest(tmp_path, {"mace": _env_record(source_hash)})
    save_manifest(manifest, tmp_path)

    state = read_install_state(tmp_path)

    env = state.envs["mace"]
    assert env.record is not None
    assert env.record.built_at == "2026-01-01T00:00:00Z"
    assert not env.source_hash_drifted


def test_source_hash_drift_detected(tmp_path):
    _make_built_env(tmp_path)
    manifest = _manifest(tmp_path, {"mace": _env_record("sha256:stale")})
    save_manifest(manifest, tmp_path)

    state = read_install_state(tmp_path)

    assert state.envs["mace"].source_hash_drifted


def test_manifest_only_envs_are_not_installed(tmp_path):
    _make_built_env(tmp_path)
    manifest = _manifest(tmp_path, {"mace": _env_record(), "deleted": _env_record()})
    save_manifest(manifest, tmp_path)

    state = read_install_state(tmp_path)

    assert set(state.envs) == {"mace"}
    assert set(state.manifest_only_envs) == {"deleted"}


def test_env_without_source_file(tmp_path):
    _make_built_env(tmp_path, source=None)

    state = read_install_state(tmp_path)

    env = state.envs["mace"]
    assert env.source_file is None
    assert env.source_hash is None
    assert env.declared_checkpoints is None


def test_malformed_checkpoints_dict_yields_none(tmp_path):
    _make_built_env(tmp_path, source="CHECKPOINTS = [1, 2]\n")

    state = read_install_state(tmp_path)

    env = state.envs["mace"]
    assert env.declared_checkpoints is None
    assert env.source_hash is not None  # hash still derivable


def test_lock_hash_read_from_disk(tmp_path):
    env_dir = _make_built_env(tmp_path)
    lock = env_dir / "env_source.py.lock"
    lock.write_text("version = 1\n")

    state = read_install_state(tmp_path)

    assert state.envs["mace"].lock_hash == compute_source_hash(lock)


def test_explicit_manifest_instance_is_aliased(tmp_path):
    """A caller about to mutate a manifest passes its own instance; the
    per-env records must alias it, not a fresh load from disk."""
    _make_built_env(tmp_path)
    manifest = _manifest(tmp_path, {"mace": _env_record()})

    state = read_install_state(tmp_path, manifest=manifest)

    assert state.manifest is manifest
    assert state.envs["mace"].record is manifest.environments["mace"]


def test_explicit_none_manifest_skips_loading(tmp_path):
    _make_built_env(tmp_path)
    save_manifest(_manifest(tmp_path, {"mace": _env_record()}), tmp_path)

    state = read_install_state(tmp_path, manifest=None)

    assert state.manifest is None
    assert state.envs["mace"].record is None


def test_sources_listed(tmp_path):
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "mace.py").write_text(ENV_SOURCE)

    state = read_install_state(tmp_path)

    assert [name for name, _ in state.sources] == ["mace"]
