"""Tests for ``rootstock status`` rendering and --json."""

from __future__ import annotations

import json
from pathlib import Path

from rootstock.commands.status import _checkpoint_line, cmd_status
from rootstock.config import UserConfig
from rootstock.manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    VerificationRecord,
    compute_source_hash,
    create_manifest,
    save_manifest,
)

ENV_SOURCE = 'CHECKPOINTS = {"mace-mp-0-small": "small", "mace-mp-0-medium": "medium"}\n'


def _make_built_env(root: Path, name: str = "mace") -> Path:
    env_dir = root / "envs" / name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(ENV_SOURCE)
    return env_dir


def _env_with_checkpoints(
    built_at: str, source_hash: str = "sha256:abc", **checkpoints
) -> EnvironmentInfo:
    return EnvironmentInfo(
        built_at=built_at,
        source_hash=source_hash,
        source="",
        python_requires=">=3.10",
        dependencies={},
        checkpoints=dict(checkpoints),
    )


def _args(root: Path, as_json: bool):
    class _Args:
        pass

    args = _Args()
    args.root = str(root)
    args.json = as_json
    return args


def test_checkpoint_line_marks_verified():
    env = _env_with_checkpoints("2026-01-01T00:00:00Z")
    ckpt = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z",
        verifications={
            "test": VerificationRecord(
                verified_at="2026-01-03T00:00:00Z",
                verified_device="cuda",
            )
        },
    )
    line = _checkpoint_line(env, "medium", ckpt, ["test"])
    assert "✓" in line
    assert "(cuda)" in line
    assert "stale" not in line


def test_checkpoint_line_marks_stale_when_env_rebuilt():
    env = _env_with_checkpoints("2026-02-01T00:00:00Z")  # newer than verify
    ckpt = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z",
        verifications={
            "test": VerificationRecord(
                verified_at="2026-01-15T00:00:00Z",
                verified_device="cuda",
            )
        },
    )
    line = _checkpoint_line(env, "medium", ckpt, ["test"])
    assert "stale" in line


def test_checkpoint_line_marks_unverified():
    env = _env_with_checkpoints("2026-01-01T00:00:00Z")
    ckpt = CheckpointInfo(fetched_at="2026-01-02T00:00:00Z")
    line = _checkpoint_line(env, "small", ckpt, ["test"])
    assert "not verified" in line
    assert "⚠" in line


def test_checkpoint_line_includes_last_error():
    env = _env_with_checkpoints("2026-01-01T00:00:00Z")
    ckpt = CheckpointInfo(
        fetched_at="2026-01-02T00:00:00Z",
        verifications={"test": VerificationRecord(last_error="verify: RuntimeError: bad")},
    )
    line = _checkpoint_line(env, "small", ckpt, ["test"])
    assert "last error" in line
    assert "RuntimeError" in line


def test_status_json_includes_verified_current(tmp_path: Path, capsys):
    """--json should add a computed verified_current bool per checkpoint."""
    env_dir = _make_built_env(tmp_path)
    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(tmp_path, ["test"], cfg)
    manifest.environments["mace"] = _env_with_checkpoints(
        "2026-01-01T00:00:00Z",
        source_hash=compute_source_hash(env_dir / "env_source.py"),
        **{
            "mace-mp-0-medium": CheckpointInfo(
                fetched_at="2026-01-02T00:00:00Z",
                verifications={
                    "test": VerificationRecord(
                        verified_at="2026-01-03T00:00:00Z",
                        verified_device="cuda",
                    )
                },
            ),
            "mace-mp-0-small": CheckpointInfo(fetched_at="2026-01-02T00:00:00Z"),
        },
    )
    save_manifest(manifest, tmp_path)

    rc = cmd_status(_args(tmp_path, as_json=True))
    assert rc == 0

    parsed = json.loads(capsys.readouterr().out)
    env = parsed["environments"]["mace"]
    assert env["in_manifest"] is True
    assert env["built_at"] == "2026-01-01T00:00:00Z"
    assert env["clusters"] is None  # unrestricted env serves every cluster
    assert env["checkpoints"]["mace-mp-0-medium"]["verified_current"] == {"test": True}
    assert env["checkpoints"]["mace-mp-0-small"]["verified_current"] == {"test": False}
    assert "verifications" in env["checkpoints"]["mace-mp-0-medium"]
    assert env["declared_checkpoints"] == ["mace-mp-0-medium", "mace-mp-0-small"]
    assert parsed["clusters"] == ["test"]


def test_status_json_no_manifest(tmp_path: Path, capsys):
    """--json without a manifest still lists what's installed on disk."""
    _make_built_env(tmp_path)

    rc = cmd_status(_args(tmp_path, as_json=True))
    assert rc == 0

    parsed = json.loads(capsys.readouterr().out)
    assert parsed["clusters"] is None
    env = parsed["environments"]["mace"]
    assert env["in_manifest"] is False
    assert env["built_at"] is None
    assert env["has_source"] is True


def test_status_json_separates_manifest_only_envs(tmp_path: Path, capsys):
    """An env recorded in the manifest but gone from disk must not be
    presented as installed."""
    _make_built_env(tmp_path)
    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(tmp_path, ["test"], cfg)
    manifest.environments["deleted"] = _env_with_checkpoints("2026-01-01T00:00:00Z")
    save_manifest(manifest, tmp_path)

    rc = cmd_status(_args(tmp_path, as_json=True))
    assert rc == 0

    parsed = json.loads(capsys.readouterr().out)
    assert "deleted" not in parsed["environments"]
    assert parsed["manifest_only_environments"] == ["deleted"]
    assert "mace" in parsed["environments"]


def test_status_human_view_flags_drift_and_manifest_only(tmp_path: Path, capsys):
    _make_built_env(tmp_path)
    cfg = UserConfig(name="t", email="t@t.t")
    manifest = create_manifest(tmp_path, ["test"], cfg)
    manifest.environments["mace"] = _env_with_checkpoints(
        "2026-01-01T00:00:00Z", source_hash="sha256:stale"
    )
    manifest.environments["deleted"] = _env_with_checkpoints("2026-01-01T00:00:00Z")
    save_manifest(manifest, tmp_path)

    rc = cmd_status(_args(tmp_path, as_json=False))
    assert rc == 0

    out = capsys.readouterr().out
    assert "differs from the manifest" in out
    assert "manifest only — not on disk" in out


def test_status_skips_cache_sizes_by_default(tmp_path: Path, capsys, monkeypatch):
    """The rglob+stat sweep is minutes on Lustre for big HF caches — it must
    not run unless asked."""
    _make_built_env(tmp_path)
    cache_dir = tmp_path / "cache" / "huggingface"
    cache_dir.mkdir(parents=True)
    (cache_dir / "weights.bin").write_bytes(b"x" * 2048)

    sweep = []
    real_rglob = Path.rglob

    def counting_rglob(self, pattern):
        sweep.append(self)
        return real_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", counting_rglob)

    rc = cmd_status(_args(tmp_path, as_json=False))
    assert rc == 0

    out = capsys.readouterr().out
    assert "--sizes" in out  # points the user at the opt-in
    assert "MB" not in out
    assert sweep == []  # no recursive sweep happened


def test_status_sizes_flag_computes_cache_sizes(tmp_path: Path, capsys):
    _make_built_env(tmp_path)
    cache_dir = tmp_path / "cache" / "huggingface"
    cache_dir.mkdir(parents=True)
    (cache_dir / "weights.bin").write_bytes(b"x" * 2048)

    args = _args(tmp_path, as_json=False)
    args.sizes = True
    rc = cmd_status(args)
    assert rc == 0

    out = capsys.readouterr().out
    assert "huggingface/" in out
    assert "MB" in out
    assert "TOTAL" in out
