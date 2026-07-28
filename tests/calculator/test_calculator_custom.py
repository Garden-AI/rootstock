"""Tests for RootstockCalculator's ':custom' checkpoint / weights= pairing."""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.calculator import RootstockCalculator
from rootstock.environment import CheckpointNotFoundError, CustomWeightsError

_UMA_ENV_SOURCE = '''\
"""UMA env."""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma:custom": None,
}


def setup(checkpoint, device="cuda", task="omat"):
    return None


def setup_from_path(path, device="cuda", task="omat"):
    return None
'''

# Authoring bug an install lint should have caught: the entry is declared but
# the hook is missing. Construction must still fail with a maintainer hint.
_ENTRY_NO_HOOK_ENV_SOURCE = """\
CHECKPOINTS = {
    "orb-v2": "orb-v2",
    "orb:custom": None,
}


def setup(checkpoint, device="cuda"):
    return None
"""

_NO_CUSTOM_ENV_SOURCE = """\
CHECKPOINTS = {"orb-v2": "orb-v2"}


def setup(checkpoint, device="cuda"):
    return None
"""


def _make_root(tmp_path: Path, env_name: str, source: str) -> Path:
    env_dir = tmp_path / "root" / "envs" / env_name
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(source)
    return tmp_path / "root"


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    return _make_root(tmp_path, "uma", _UMA_ENV_SOURCE)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "ft.pt"
    path.write_bytes(b"weights")
    return path


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


def test_custom_id_binds_weights_as_checkpoint_path(fake_root, weights, recording_server):
    calc = RootstockCalculator(
        checkpoint="uma:custom", root=fake_root, weights=weights, device="cpu"
    )
    assert calc.env_name == "uma"
    assert calc.checkpoint_path == str(weights)
    calc._ensure_server()
    kwargs = recording_server.ctor_kwargs
    assert kwargs["checkpoint_path"] == str(weights)
    assert kwargs["checkpoint"] == "uma:custom"


def test_relative_weights_path_resolved_at_construction(
    fake_root, weights, recording_server, monkeypatch
):
    """The worker runs with cwd=env_dir, so a relative path passed through
    verbatim would be re-resolved there — it must leave construction
    absolute."""
    monkeypatch.chdir(weights.parent)
    calc = RootstockCalculator(
        checkpoint="uma:custom", root=fake_root, weights=weights.name, device="cpu"
    )
    assert calc.checkpoint_path == str(weights.resolve())


def test_custom_without_weights_raises(fake_root):
    with pytest.raises(CustomWeightsError, match="weights"):
        RootstockCalculator(checkpoint="uma:custom", root=fake_root)


def test_weights_without_custom_names_the_entry(fake_root, weights):
    # The canonical id resolves to an env, and the env declares its ':custom'
    # entries — so the error can name the exact id to use.
    with pytest.raises(CustomWeightsError, match="uma:custom"):
        RootstockCalculator(checkpoint="uma-s-1p1", root=fake_root, weights=weights)


def test_weights_without_custom_no_entry_points_at_maintainer(tmp_path, weights):
    root = _make_root(tmp_path, "orb", _NO_CUSTOM_ENV_SOURCE)
    with pytest.raises(CustomWeightsError, match="maintainer"):
        RootstockCalculator(checkpoint="orb-v2", root=root, weights=weights)


def test_misspelled_weight_kwarg_raises_not_silently_absorbed(fake_root, weights):
    """The ASE-absorption regression: Calculator.__init__ quietly accepts
    unknown kwargs, so a misspelled weight= must still fail via the ':custom'
    pairing guard — with a custom id there are no shipped weights to
    silently run instead."""
    with pytest.raises(CustomWeightsError):
        RootstockCalculator(checkpoint="uma:custom", root=fake_root, weight=str(weights))


def test_missing_weights_file_raises_at_construction(fake_root, weights, recording_server):
    weights.unlink()
    with pytest.raises(CustomWeightsError, match="not found"):
        RootstockCalculator(checkpoint="uma:custom", root=fake_root, weights=weights)
    assert recording_server.ctor_kwargs == {}


def test_hookless_env_raises_at_construction(tmp_path, weights, recording_server):
    """A built env whose entry lacks the setup_from_path hook must fail here
    with a maintainer hint, not as an ImportError inside the worker ->
    opaque WorkerDiedError."""
    root = _make_root(tmp_path, "orb", _ENTRY_NO_HOOK_ENV_SOURCE)
    with pytest.raises(CustomWeightsError, match="maintainer"):
        RootstockCalculator(checkpoint="orb:custom", root=root, weights=weights)
    assert recording_server.ctor_kwargs == {}


def test_path_setup_kwarg_rejected_for_custom(fake_root, weights):
    # "path" is setup_from_path's first parameter; fail at construction, not
    # as a TypeError inside the worker.
    with pytest.raises(CustomWeightsError, match="path"):
        RootstockCalculator(
            checkpoint="uma:custom",
            root=fake_root,
            weights=weights,
            setup_kwargs={"path": "/x"},
        )


def test_undeclared_custom_id_raises_not_found(fake_root, weights):
    with pytest.raises(CheckpointNotFoundError, match="umma:custom"):
        RootstockCalculator(checkpoint="umma:custom", root=fake_root, weights=weights)


def test_canonical_id_unaffected(fake_root, recording_server):
    calc = RootstockCalculator(checkpoint="uma-s-1p1", root=fake_root, device="cpu")
    assert calc.checkpoint_path is None
    calc._ensure_server()
    assert recording_server.ctor_kwargs["checkpoint_path"] is None
