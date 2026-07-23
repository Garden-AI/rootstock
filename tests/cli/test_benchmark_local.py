"""Tests for benchmark guards on local (user-registered) checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock import local_checkpoints
from rootstock.benchmark import benchmark_one
from rootstock.local_checkpoints import register_local_checkpoint

_ENV_SOURCE = """\
CHECKPOINTS = {"uma-s-1p1": "uma-s-1p1"}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    env_dir = tmp_path / "root" / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE)
    return tmp_path / "root"


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "registry.json"
    monkeypatch.setattr(local_checkpoints, "DEFAULT_LOCAL_REGISTRY_FILE", path)
    return path


def test_benchmark_one_missing_local_weights(fake_root, registry, tmp_path):
    # Fails before either arm spawns — same message as the calculator guard —
    # instead of a raw subprocess traceback from the in-env worker.
    weights = tmp_path / "ft.pt"
    weights.write_bytes(b"weights")
    register_local_checkpoint(fake_root, "my-ft", "uma", weights)
    weights.unlink()

    with pytest.raises(RuntimeError, match="no longer exists"):
        benchmark_one(
            checkpoint="my-ft",
            device="cpu",
            root=fake_root,
            cache_root=None,
            cluster=None,
            atoms=None,
            frames=None,
            n_warmup=0,
            setup_kwargs={},
            work_dir=tmp_path,
        )
