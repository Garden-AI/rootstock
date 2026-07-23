"""Tests for RootstockCalculator with local (user-registered) checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock import local_checkpoints
from rootstock.calculator import RootstockCalculator
from rootstock.environment import CheckpointNotFoundError
from rootstock.local_checkpoints import LocalCheckpointError, register_local_checkpoint

_UMA_ENV_SOURCE = '''\
"""UMA env."""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
}


def setup(checkpoint, device="cuda", task="omat"):
    return None


def setup_from_path(path, device="cuda", task="omat"):
    return None
'''


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    env_dir = tmp_path / "root" / "envs" / "uma"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_UMA_ENV_SOURCE)
    return tmp_path / "root"


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> Path:
    # The calculator resolves against the module default; point it into tmp
    # so tests never touch (or depend on) the developer's real registry.
    path = tmp_path / "registry.json"
    monkeypatch.setattr(local_checkpoints, "DEFAULT_LOCAL_REGISTRY_FILE", path)
    return path


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "ft.pt"
    path.write_bytes(b"weights")
    return path


@pytest.fixture
def registered(fake_root, weights, registry) -> str:
    register_local_checkpoint(
        fake_root, "my-uma-ft", "uma", weights, setup_kwargs={"task": "omol"}
    )
    return "my-uma-ft"


class _RecordingServer:
    ctor_kwargs: dict = {}

    def __init__(self, **kwargs):
        _RecordingServer.ctor_kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def recording_server(monkeypatch):
    _RecordingServer.ctor_kwargs = {}
    monkeypatch.setattr("rootstock.calculator.RootstockServer", _RecordingServer)
    return _RecordingServer


def test_local_id_resolves_env_and_path(fake_root, registered, weights):
    calc = RootstockCalculator(checkpoint=registered, root=fake_root)
    assert calc.env_name == "uma"
    assert calc.checkpoint_path == str(weights.resolve())
    assert calc.setup_kwargs == {"task": "omol"}


def test_per_call_kwargs_override_registered(fake_root, registered):
    calc = RootstockCalculator(
        checkpoint=registered, root=fake_root, setup_kwargs={"task": "omc"}
    )
    assert calc.setup_kwargs == {"task": "omc"}


def test_server_receives_checkpoint_path(fake_root, registered, weights, recording_server):
    calc = RootstockCalculator(checkpoint=registered, root=fake_root, device="cpu")
    calc._ensure_server()
    kwargs = recording_server.ctor_kwargs
    assert kwargs["checkpoint_path"] == str(weights.resolve())
    assert kwargs["checkpoint"] == registered
    assert kwargs["setup_kwargs"] == {"task": "omol"}


def test_canonical_id_has_no_checkpoint_path(fake_root, registry, recording_server):
    calc = RootstockCalculator(checkpoint="uma-s-1p1", root=fake_root, device="cpu")
    assert calc.checkpoint_path is None
    calc._ensure_server()
    assert recording_server.ctor_kwargs["checkpoint_path"] is None


def test_per_call_path_kwarg_rejected_for_local(fake_root, registered):
    # "path" is setup_from_path's first parameter; fail at construction, not
    # as a TypeError inside the worker.
    with pytest.raises(TypeError, match="path"):
        RootstockCalculator(
            checkpoint=registered, root=fake_root, setup_kwargs={"path": "/x"}
        )


def test_missing_weights_file_raises_at_construction(
    fake_root, registered, weights, recording_server
):
    weights.unlink()
    with pytest.raises(LocalCheckpointError, match="no longer exists"):
        RootstockCalculator(checkpoint=registered, root=fake_root)
    # Never got as far as building a server.
    assert recording_server.ctor_kwargs == {}


def test_unknown_id_mentions_add_local(fake_root, registry):
    with pytest.raises(CheckpointNotFoundError, match="add-local"):
        RootstockCalculator(checkpoint="not-a-real-id", root=fake_root)
