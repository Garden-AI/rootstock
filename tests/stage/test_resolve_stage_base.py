"""The staging-base resolution chain: env var > layout.json > registry,
with validation that fails closed (disabled) rather than staging badly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rootstock.stage as stage
from rootstock.clusters import Cluster
from rootstock.stage import NO_STAGE_ENV, STAGE_DIR_ENV, resolve_stage_base


@pytest.fixture
def different_fs(monkeypatch):
    """tmp_path trees share one device; pretend base and root differ."""
    monkeypatch.setattr(stage, "_same_filesystem", lambda a, b: False)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    return root


def _declare_layout(root: Path, stage_dir: str) -> None:
    (root / "layout.json").write_text(json.dumps({"layout_version": 1, "stage_dir": stage_dir}))


def test_env_var_wins(monkeypatch, root: Path, tmp_path: Path, different_fs):
    env_base = tmp_path / "from-env"
    env_base.mkdir()
    layout_base = tmp_path / "from-layout"
    layout_base.mkdir()
    _declare_layout(root, str(layout_base))
    monkeypatch.setenv(STAGE_DIR_ENV, str(env_base))
    assert resolve_stage_base(root) == env_base


def test_layout_declaration(monkeypatch, root: Path, tmp_path: Path, different_fs):
    monkeypatch.delenv(STAGE_DIR_ENV, raising=False)
    base = tmp_path / "local"
    base.mkdir()
    _declare_layout(root, str(base))
    assert resolve_stage_base(root) == base


def test_registry_fallback(monkeypatch, root: Path, tmp_path: Path, different_fs):
    monkeypatch.delenv(STAGE_DIR_ENV, raising=False)
    base = tmp_path / "registry-local"
    base.mkdir()
    monkeypatch.setattr(
        "rootstock.clusters.CLUSTER_REGISTRY",
        {"testcluster": Cluster(root=root, stage_dir=str(base))},
    )
    assert resolve_stage_base(root) == base


def test_nothing_declared_disables(monkeypatch, root: Path, different_fs):
    monkeypatch.delenv(STAGE_DIR_ENV, raising=False)
    assert resolve_stage_base(root) is None


def test_no_stage_env_force_disables(monkeypatch, root: Path, tmp_path: Path, different_fs):
    base = tmp_path / "local"
    base.mkdir()
    monkeypatch.setenv(STAGE_DIR_ENV, str(base))
    monkeypatch.setenv(NO_STAGE_ENV, "1")
    assert resolve_stage_base(root) is None


def test_unexpanded_env_var_disables(monkeypatch, root: Path, different_fs):
    # $SLURM_TMPDIR outside a job: the declaration is fine, this node isn't.
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv(STAGE_DIR_ENV, "$SLURM_TMPDIR/stage")
    assert resolve_stage_base(root) is None


def test_env_var_expansion(monkeypatch, root: Path, tmp_path: Path, different_fs):
    base = tmp_path / "jobtmp"
    base.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(base))
    monkeypatch.setenv(STAGE_DIR_ENV, "$SLURM_TMPDIR")
    assert resolve_stage_base(root) == base


def test_missing_dir_disables(monkeypatch, root: Path, tmp_path: Path, different_fs):
    monkeypatch.setenv(STAGE_DIR_ENV, str(tmp_path / "nope"))
    assert resolve_stage_base(root) is None


def test_same_filesystem_disables(monkeypatch, root: Path, tmp_path: Path):
    # No _same_filesystem patch here: base and root genuinely share a device,
    # and staging onto the filesystem we're escaping must be refused.
    base = tmp_path / "samefs"
    base.mkdir()
    monkeypatch.setenv(STAGE_DIR_ENV, str(base))
    assert resolve_stage_base(root) is None
