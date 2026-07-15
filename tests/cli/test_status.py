"""Tests for ``rootstock status`` rendering and --json."""

from __future__ import annotations

import json
from pathlib import Path

from rootstock.commands.status import _checkpoint_line, cmd_status
from rootstock.config import UserConfig
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    create_manifest,
    save_manifest,
)


def _env_with_checkpoints(built_at: str, **checkpoints) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at=built_at,
        source_hash="sha256:abc",
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints=dict(checkpoints),
    )


def test_checkpoint_line_marks_verified():
    env = _env_with_checkpoints("2026-01-01T00:00:00Z")
    ckpt = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z",
        verified_at="2026-01-03T00:00:00Z",
        verified_device="cuda",
    )
    line = _checkpoint_line(env, "medium", ckpt)
    assert "✓" in line
    assert "(cuda)" in line
    assert "stale" not in line


def test_checkpoint_line_marks_stale_when_env_rebuilt():
    env = _env_with_checkpoints("2026-02-01T00:00:00Z")  # newer than verify
    ckpt = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z",
        verified_at="2026-01-15T00:00:00Z",
        verified_device="cuda",
    )
    line = _checkpoint_line(env, "medium", ckpt)
    assert "stale" in line


def test_checkpoint_line_marks_unverified():
    env = _env_with_checkpoints("2026-01-01T00:00:00Z")
    ckpt = CheckpointInfo(fetched_at="2026-01-02T00:00:00Z")
    line = _checkpoint_line(env, "small", ckpt)
    assert "not verified" in line
    assert "⚠" in line


def test_checkpoint_line_includes_last_error():
    env = _env_with_checkpoints("2026-01-01T00:00:00Z")
    ckpt = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z",
        last_error="verify: RuntimeError: bad",
    )
    line = _checkpoint_line(env, "small", ckpt)
    assert "last error" in line
    assert "RuntimeError" in line


def test_status_json_includes_verified_current(tmp_path: Path, capsys):
    """--json should add a computed verified_current bool per checkpoint."""
    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(tmp_path, "test", cfg)
    manifest.environments["mace"] = _env_with_checkpoints(
        "2026-01-01T00:00:00Z",
        **{
            "mace-mp-0-medium": CheckpointInfo(
                fetched_at="2026-01-02T00:00:00Z",
                verified_at="2026-01-03T00:00:00Z",
                verified_device="cuda",
            ),
            "mace-mp-0-small": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
        },
    )
    save_manifest(manifest, tmp_path)

    class _Args:
        pass

    args = _Args()
    args.root = str(tmp_path)
    args.json = True

    rc = cmd_status(args)
    assert rc == 0

    parsed = json.loads(capsys.readouterr().out)
    ckpts = parsed["manifest"]["environments"]["mace"]["checkpoints"]
    assert ckpts["mace-mp-0-medium"]["verified_current"] is True
    assert ckpts["mace-mp-0-small"]["verified_current"] is False


def test_status_json_no_manifest(tmp_path: Path, capsys):
    """--json without a manifest should still emit valid JSON."""

    class _Args:
        pass

    args = _Args()
    args.root = str(tmp_path)
    args.json = True

    rc = cmd_status(args)
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["manifest"] is None
