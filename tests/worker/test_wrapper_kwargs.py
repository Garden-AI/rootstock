"""Tests for setup_kwargs plumbing through the wrapper script."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from rootstock.environment import EnvironmentManager


def _extract_kwargs_path(wrapper_text: str) -> Path:
    match = re.search(r'open\("([^"]+)"\)', wrapper_text)
    assert match, f"could not find kwargs_path in wrapper: {wrapper_text!r}"
    return Path(match.group(1))


@pytest.fixture
def env_manager(tmp_path: Path) -> EnvironmentManager:
    # Pretend an env is "built" — generate_wrapper doesn't actually exec it.
    (tmp_path / "envs" / "fake_env").mkdir(parents=True)
    mgr = EnvironmentManager(root=tmp_path)
    yield mgr
    mgr.cleanup()


def test_wrapper_writes_empty_kwargs_sidecar_when_none(env_manager):
    wrapper = env_manager.generate_wrapper(
        env_name="fake_env",
        model="m",
        device="cpu",
        socket_path="/tmp/sock",
    )
    kwargs_path = _extract_kwargs_path(wrapper.read_text())
    assert json.loads(kwargs_path.read_text()) == {}


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
def test_wrapper_round_trips_kwargs_through_json_sidecar(env_manager, kwargs):
    wrapper = env_manager.generate_wrapper(
        env_name="fake_env",
        model="m",
        device="cpu",
        socket_path="/tmp/sock",
        setup_kwargs=kwargs,
    )
    kwargs_path = _extract_kwargs_path(wrapper.read_text())
    assert json.loads(kwargs_path.read_text()) == kwargs


def test_wrapper_and_kwargs_files_cleaned_up(env_manager):
    wrapper = env_manager.generate_wrapper(
        env_name="fake_env",
        model="m",
        device="cpu",
        socket_path="/tmp/sock",
        setup_kwargs={"task": "omol"},
    )
    kwargs_path = _extract_kwargs_path(wrapper.read_text())
    assert wrapper.exists()
    assert kwargs_path.exists()

    env_manager.cleanup()
    assert not wrapper.exists()
    assert not kwargs_path.exists()


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
def fake_setup(model, device, **kwargs):
    with open({str(record_path)!r}, "w") as f:
        json.dump({{"model": model, "device": device, "kwargs": kwargs}}, f)
    raise SystemExit(0)

from rootstock.worker import run_worker
try:
    run_worker(
        setup_fn=fake_setup,
        model="checkpoint-x",
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
        "model": "checkpoint-x",
        "device": "cpu",
        "kwargs": {"task": "omol", "charge": -1},
    }
