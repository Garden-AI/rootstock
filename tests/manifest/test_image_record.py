"""The per-env packed-image record (#180): round-trip, the packed_at >=
built_at currency check, and how the manifest refresh preserves/stamps it."""

from __future__ import annotations

from pathlib import Path

from rootstock.manifest import (
    SCHEMA_VERSION,
    EnvironmentInfo,
    Maintainer,
    Manifest,
    image_is_current,
)
from rootstock.operations import refresh_manifest_environments

ENV_SOURCE = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    '# dependencies = ["six>=1.0"]\n'
    "# ///\n"
    "CHECKPOINTS = {}\n"
)

IMAGE = {
    "path": "images/demo-abc123def456.tar.zst",
    "sha256": "abc123def456" + "0" * 52,
    "format": "tar.zst",
    "compressed_bytes": 100,
    "uncompressed_bytes": 250,
}


def _env(built_at: str, image: dict | None) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at=built_at,
        source_hash=None,
        source="",
        python_requires=">=3.10",
        dependencies={},
        image=image,
    )


def test_image_round_trips_through_dataclasses():
    env = _env("2026-09-01T00:00:00Z", {**IMAGE, "packed_at": "2026-09-01T00:00:00Z"})
    assert EnvironmentInfo.from_dict(env.to_dict()).image == env.image


def test_image_currency():
    built = "2026-09-01T00:00:00Z"
    assert image_is_current(_env(built, {**IMAGE, "packed_at": built}))  # same refresh
    assert image_is_current(_env(built, {**IMAGE, "packed_at": "2026-09-02T00:00:00Z"}))
    # Rebuilt after the pack: the record survives but must not read as usable.
    assert not image_is_current(_env("2026-09-03T00:00:00Z", {**IMAGE, "packed_at": built}))
    assert not image_is_current(_env(built, None))
    assert not image_is_current(_env(built, IMAGE))  # no packed_at at all


def _make_built_env(root: Path, name: str) -> None:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(ENV_SOURCE)


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


def test_refresh_preserves_existing_image_record(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rootstock.operations.get_installed_versions", lambda *a, **k: {})
    _make_built_env(tmp_path, "demo")
    existing = _manifest(
        tmp_path,
        {"demo": _env("2026-09-01T00:00:00Z", {**IMAGE, "packed_at": "2026-09-01T00:00:00Z"})},
    )

    refreshed = refresh_manifest_environments(existing, tmp_path)

    assert refreshed.environments["demo"].image == {**IMAGE, "packed_at": "2026-09-01T00:00:00Z"}


def test_refresh_stamps_freshly_packed_image_current(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("rootstock.operations.get_installed_versions", lambda *a, **k: {})
    _make_built_env(tmp_path, "demo")

    refreshed = refresh_manifest_environments(
        _manifest(tmp_path), tmp_path, built_env="demo", packed_images={"demo": IMAGE}
    )

    env = refreshed.environments["demo"]
    assert env.image is not None and "packed_at" in env.image
    # Stamped in the same refresh as built_at: the currency check must hold
    # for an install's own pack, or every fresh install would fall back.
    assert image_is_current(env)
