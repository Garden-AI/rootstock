"""Tests for benchmark guards on ':custom' checkpoints (--weights)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.benchmark import benchmark_one, main
from rootstock.environment import CustomWeightsError

_ENV_SOURCE = """\
CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma:custom": None,
}


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


def _run_one(fake_root, tmp_path, checkpoint, weights):
    return benchmark_one(
        checkpoint=checkpoint,
        device="cpu",
        root=fake_root,
        cache_root=None,
        cluster=None,
        atoms=None,
        frames=None,
        n_warmup=0,
        setup_kwargs={},
        work_dir=tmp_path,
        weights=weights,
    )


def test_benchmark_one_custom_without_weights(fake_root, tmp_path):
    with pytest.raises(CustomWeightsError, match="weights"):
        _run_one(fake_root, tmp_path, "uma:custom", None)


def test_benchmark_one_weights_without_custom(fake_root, tmp_path):
    weights = tmp_path / "ft.pt"
    weights.write_bytes(b"weights")
    with pytest.raises(CustomWeightsError, match="uma:custom"):
        _run_one(fake_root, tmp_path, "uma-s-1p1", str(weights))


def test_benchmark_one_custom_missing_weights_file(fake_root, tmp_path):
    # Fails before either arm spawns, not as a raw subprocess traceback.
    with pytest.raises(CustomWeightsError, match="not found"):
        _run_one(fake_root, tmp_path, "uma:custom", str(tmp_path / "gone.pt"))


def test_parser_accepts_weights_flag(fake_root, capsys):
    # --weights parses on the public parser; a guard failure per checkpoint
    # is recorded as an error row instead of aborting the run.
    rc = main(
        [
            "--root",
            str(fake_root),
            "--checkpoints",
            "uma:custom",
            "--weights",
            str(fake_root / "gone.pt"),
            "--calls",
            "1",
            "--warmup",
            "0",
        ]
    )
    assert rc == 0
    assert "ERROR" in capsys.readouterr().out
