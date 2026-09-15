"""How staging plugs into spawn_in_env: the sidecar, interpreter, cache env
vars, and prewarm toggle all follow the staged copy — and downloads and
capture runs never see a mirror."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock.spawn import DOWNLOAD_WRAPPER, WORKER_WRAPPER, spawn_in_env
from rootstock.stage import StagedSpawn, stage_for_spawn


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    bin_dir = root / "envs" / "demo" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("#!/bin/sh\n")
    return root


@pytest.fixture
def staged(tmp_path: Path) -> StagedSpawn:
    staged_env = tmp_path / "local" / "sha" / "envs" / "demo"
    (staged_env / "bin").mkdir(parents=True)
    (staged_env / "bin" / "python").write_text("#!/bin/sh\n")
    mirror = tmp_path / "local" / "cache-mirror"
    mirror.mkdir(parents=True)
    return StagedSpawn(env_dir=staged_env, cache_base=mirror)


def _sidecar(spec) -> dict:
    return json.loads(Path(spec.cmd[2]).read_text())


def test_staged_spawn_is_fully_local(monkeypatch, root: Path, staged: StagedSpawn):
    monkeypatch.setattr("rootstock.stage.stage_for_spawn", lambda *a, **k: staged)
    payload = {"checkpoint": "demo-ckpt", "device": "cpu", "setup_kwargs": {}}
    with spawn_in_env(root, "demo", WORKER_WRAPPER, payload) as spec:
        assert spec.cmd[0] == str(staged.env_dir / "bin" / "python")
        assert spec.cwd == str(staged.env_dir)
        assert _sidecar(spec)["env_dir"] == str(staged.env_dir)
        # caches point at the mirror, prewarm is redundant
        assert spec.env["HOME"] == str(staged.cache_base / "home")
        assert spec.env["XDG_CACHE_HOME"] == str(staged.cache_base / "cache")
        assert spec.env["ROOTSTOCK_NO_PREWARM"] == "1"


def test_staged_env_without_weights_keeps_prewarm(monkeypatch, root: Path, staged: StagedSpawn):
    staged.cache_base = None
    monkeypatch.setattr("rootstock.stage.stage_for_spawn", lambda *a, **k: staged)
    payload = {"checkpoint": "demo-ckpt", "device": "cpu", "setup_kwargs": {}}
    with spawn_in_env(root, "demo", WORKER_WRAPPER, payload) as spec:
        assert spec.cmd[0] == str(staged.env_dir / "bin" / "python")
        # caches stay on the shared filesystem, and the prewarm still runs
        # (streaming the shared weights ahead of the mmaps).
        assert spec.env["HOME"] == str(root / "home")
        assert "ROOTSTOCK_NO_PREWARM" not in spec.env


def test_custom_weights_keep_prewarm_even_when_staged(monkeypatch, root: Path, staged: StagedSpawn):
    monkeypatch.setattr("rootstock.stage.stage_for_spawn", lambda *a, **k: staged)
    payload = {
        "checkpoint": "demo:custom",
        "checkpoint_path": "/scratch/me/ft.pt",
        "device": "cpu",
        "setup_kwargs": {},
    }
    with spawn_in_env(root, "demo", WORKER_WRAPPER, payload) as spec:
        # The user's weights file still lives on the shared filesystem; the
        # prewarm must stay on to stream it.
        assert "ROOTSTOCK_NO_PREWARM" not in spec.env


def test_download_spawns_never_stage(monkeypatch, root: Path):
    def boom(*a, **k):  # pragma: no cover - the assertion is that it's unused
        raise AssertionError("download spawns must not stage")

    monkeypatch.setattr("rootstock.stage.stage_for_spawn", boom)
    payload = {"checkpoint": "demo-ckpt", "device": "cpu", "setup_kwargs": {}}
    with spawn_in_env(root, "demo", DOWNLOAD_WRAPPER, payload) as spec:
        assert spec.cmd[0] == str(root / "envs" / "demo" / "bin" / "python")


def test_unstaged_spawn_unchanged(monkeypatch, root: Path):
    monkeypatch.setattr("rootstock.stage.stage_for_spawn", lambda *a, **k: None)
    payload = {"checkpoint": "demo-ckpt", "device": "cpu", "setup_kwargs": {}}
    with spawn_in_env(root, "demo", WORKER_WRAPPER, payload) as spec:
        assert spec.cmd[0] == str(root / "envs" / "demo" / "bin" / "python")
        assert spec.env["HOME"] == str(root / "home")
        assert "ROOTSTOCK_NO_PREWARM" not in spec.env


def test_capture_spawns_stage_env_but_not_weights(monkeypatch, tmp_path: Path):
    """Verify/add runs (weights_capture in the payload) may stage the env,
    but must observe the shared cache — never a mirror."""
    staged_root = tmp_path / "sha"

    def fake_stage_weights(*a, **k):  # pragma: no cover
        raise AssertionError("capture spawns must not stage weights")

    monkeypatch.setattr("rootstock.stage.resolve_stage_base", lambda root: tmp_path)
    monkeypatch.setattr("rootstock.stage.stage_env", lambda *a, **k: staged_root)
    monkeypatch.setattr("rootstock.stage.stage_weights", fake_stage_weights)

    payload = {
        "checkpoint": "demo-ckpt",
        "weights_capture": {"result_path": "/tmp/x", "cache_root": "/shared"},
    }
    staged = stage_for_spawn(tmp_path / "root", "demo", payload)
    assert staged is not None
    assert staged.env_dir == staged_root / "envs" / "demo"
    assert staged.cache_base is None
