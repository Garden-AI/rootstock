"""End-to-end tests of WORKER_WRAPPER's setup vs setup_from_path branch.

Executes the actual wrapper text under this interpreter against a fake env
whose loaders record their arguments and exit before any socket I/O.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rootstock.spawn import WORKER_WRAPPER

_ENV_SOURCE_TEMPLATE = """\
import json

CHECKPOINTS = {{"canon-1": "canon"}}


def setup(checkpoint, device="cuda", **kwargs):
    with open({record!r}, "w") as f:
        json.dump({{"fn": "setup", "target": checkpoint, "device": device, "kwargs": kwargs}}, f)
    raise SystemExit(0)


def setup_from_path(path, device="cuda", **kwargs):
    with open({record!r}, "w") as f:
        json.dump(
            {{"fn": "setup_from_path", "target": path, "device": device, "kwargs": kwargs}}, f
        )
    raise SystemExit(0)
"""


@pytest.fixture
def staged(tmp_path: Path):
    """Stage the real wrapper text + a fake env_source; return a runner."""
    record = tmp_path / "record.json"
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE_TEMPLATE.format(record=str(record)))
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(WORKER_WRAPPER)

    def run(spec: dict) -> dict:
        sidecar = tmp_path / "spec.json"
        sidecar.write_text(json.dumps({**spec, "env_dir": str(env_dir)}))
        result = subprocess.run(
            [sys.executable, str(wrapper), str(sidecar)],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(record.read_text())

    return run


def _spec(**extra) -> dict:
    return {
        "checkpoint": "canon-1",
        "device": "cpu",
        "socket_path": "/tmp/does_not_matter",
        "setup_kwargs": {"task": "omol"},
        **extra,
    }


def test_canonical_mode_uses_setup(staged):
    record = staged(_spec())
    assert record["fn"] == "setup"
    assert record["target"] == "canon-1"
    assert record["kwargs"] == {"task": "omol"}


def test_local_mode_uses_setup_from_path(staged):
    record = staged(_spec(checkpoint="my-ft", checkpoint_path="/scratch/me/ft.pt"))
    assert record["fn"] == "setup_from_path"
    assert record["target"] == "/scratch/me/ft.pt"
    assert record["kwargs"] == {"task": "omol"}


def test_null_checkpoint_path_is_canonical(staged):
    # The server always includes the key; null must mean canonical.
    record = staged(_spec(checkpoint_path=None))
    assert record["fn"] == "setup"
    assert record["target"] == "canon-1"
