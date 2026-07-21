"""Tests for value plumbing through the spawn sidecar."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rootstock.spawn import WORKER_WRAPPER, spawn_in_env


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    # Pretend an env is "built" — spawn_in_env stages files, it doesn't exec.
    (tmp_path / "envs" / "fake_env" / "bin").mkdir(parents=True)
    (tmp_path / "envs" / "fake_env" / "bin" / "python").touch()
    return tmp_path


def _payload(**setup_kwargs) -> dict:
    return {
        "checkpoint": "m",
        "device": "cpu",
        "socket_path": "/tmp/sock",
        "setup_kwargs": setup_kwargs,
    }


def test_wrapper_source_is_static(fake_root):
    """No runtime value is ever interpolated into Python source — everything
    travels through the sidecar."""
    with spawn_in_env(fake_root, "fake_env", WORKER_WRAPPER, _payload(task="omol")) as spec:
        env_python, wrapper, sidecar = spec.cmd
        assert Path(wrapper).read_text() == WORKER_WRAPPER
        spec_data = json.loads(Path(sidecar).read_text())
        assert spec_data["checkpoint"] == "m"
        assert spec_data["device"] == "cpu"
        assert spec_data["socket_path"] == "/tmp/sock"
        assert spec_data["env_dir"] == str(fake_root / "envs" / "fake_env")


def test_empty_setup_kwargs_round_trip(fake_root):
    with spawn_in_env(fake_root, "fake_env", WORKER_WRAPPER, _payload()) as spec:
        spec_data = json.loads(Path(spec.cmd[2]).read_text())
        assert spec_data["setup_kwargs"] == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "omat"},
        {"charge": -1, "spin": 2},
        {"enabled": True, "disabled": False},
        {"nested": {"a": 1, "b": [1, 2, 3]}},
        {"unicode": "ωμα", "quote": 'has "quotes" inside'},
    ],
)
def test_setup_kwargs_round_trip_through_json_sidecar(fake_root, kwargs):
    with spawn_in_env(fake_root, "fake_env", WORKER_WRAPPER, _payload(**kwargs)) as spec:
        spec_data = json.loads(Path(spec.cmd[2]).read_text())
        assert spec_data["setup_kwargs"] == kwargs


def test_staged_files_removed_on_exit(fake_root):
    with spawn_in_env(fake_root, "fake_env", WORKER_WRAPPER, _payload()) as spec:
        wrapper, sidecar = Path(spec.cmd[1]), Path(spec.cmd[2])
        assert wrapper.exists()
        assert sidecar.exists()
    assert not wrapper.parent.exists()


def test_staged_files_removed_on_exception(fake_root):
    with pytest.raises(RuntimeError, match="boom"):
        with spawn_in_env(fake_root, "fake_env", WORKER_WRAPPER, _payload()) as spec:
            wrapper = Path(spec.cmd[1])
            raise RuntimeError("boom")
    assert not wrapper.parent.exists()


def test_unbuilt_env_raises_before_staging(fake_root):
    with pytest.raises(RuntimeError, match="not built"):
        with spawn_in_env(fake_root, "missing_env", WORKER_WRAPPER, _payload()):
            pass


def test_run_worker_forwards_setup_kwargs(tmp_path: Path):
    """End-to-end: the worker actually unpacks setup_kwargs into setup()."""
    record_path = tmp_path / "record.json"
    script = tmp_path / "drive_worker.py"
    script.write_text(
        f"""
import json, sys
from unittest.mock import patch

# Capture what setup_fn was called with, then short-circuit before the
# socket connection by raising. We just want to assert kwargs forwarding.
def fake_setup(checkpoint, device, **kwargs):
    with open({str(record_path)!r}, "w") as f:
        json.dump({{"checkpoint": checkpoint, "device": device, "kwargs": kwargs}}, f)
    raise SystemExit(0)

from rootstock.worker import run_worker
try:
    run_worker(
        setup_fn=fake_setup,
        checkpoint="mace-mp-0-medium",
        device="cpu",
        socket_path="/tmp/does_not_matter",
        setup_kwargs={{"task": "omol", "charge": -1}},
    )
except SystemExit:
    pass
"""
    )
    rc = subprocess.run([sys.executable, str(script)], cwd=Path(__file__).resolve().parents[2])
    assert rc.returncode == 0
    record = json.loads(record_path.read_text())
    assert record == {
        "checkpoint": "mace-mp-0-medium",
        "device": "cpu",
        "kwargs": {"task": "omol", "charge": -1},
    }
