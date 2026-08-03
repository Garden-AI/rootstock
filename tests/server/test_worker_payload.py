"""Tests for the worker-spec payload RootstockServer hands to spawn_in_env."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from rootstock.server import RootstockServer


class _StopSpawn(RuntimeError):
    """Raised by the fake spawn_in_env so _start_worker never launches."""


@pytest.fixture
def captured_payload(monkeypatch):
    captured = {}

    @contextmanager
    def fake_spawn(root, env_name, wrapper_source, payload, cache_root=None, offline=False):
        captured.update(payload)
        captured["_offline"] = offline
        raise _StopSpawn("payload captured")
        yield  # pragma: no cover

    # _start_worker imports spawn_in_env from .spawn at call time.
    monkeypatch.setattr("rootstock.spawn.spawn_in_env", fake_spawn)
    return captured


def _start(server) -> None:
    server.socket_path = "/tmp/fake_sock"
    with pytest.raises(_StopSpawn):
        server._start_worker()


def test_payload_carries_checkpoint_path(captured_payload, tmp_path: Path):
    server = RootstockServer(
        env_name="uma",
        checkpoint="my-ft",
        device="cpu",
        root=tmp_path,
        checkpoint_path="/scratch/me/ft.pt",
    )
    _start(server)
    assert captured_payload["checkpoint"] == "my-ft"
    assert captured_payload["checkpoint_path"] == "/scratch/me/ft.pt"


def test_payload_checkpoint_path_defaults_to_none(captured_payload, tmp_path: Path):
    server = RootstockServer(
        env_name="mace",
        checkpoint="mace-mp-0-medium",
        device="cpu",
        root=tmp_path,
    )
    _start(server)
    assert captured_payload["checkpoint_path"] is None


def test_worker_spawns_offline(captured_payload, tmp_path: Path):
    """Workers serve pre-fetched weights — hub network access is forbidden."""
    server = RootstockServer(
        env_name="uma",
        checkpoint="uma-s-1p1",
        device="cpu",
        root=tmp_path,
    )
    _start(server)
    assert captured_payload["_offline"] is True


def test_payload_omits_weights_capture_by_default(captured_payload, tmp_path: Path):
    """Regular calculator spawns never capture — the wrapper must see no key."""
    server = RootstockServer(
        env_name="mace",
        checkpoint="mace-mp-0-medium",
        device="cpu",
        root=tmp_path,
    )
    _start(server)
    assert "weights_capture" not in captured_payload


def test_payload_carries_weights_capture_with_cache_root_base(captured_payload, tmp_path: Path):
    """Verification asks for capture; records are relative to the cache root
    (the split-cache one when the install declares it, else the install root)."""
    server = RootstockServer(
        env_name="uma",
        checkpoint="uma-s-1p1",
        device="cpu",
        root=tmp_path,
        cache_root=tmp_path / "split-cache",
        weights_capture_path="/tmp/fake_weights.json",
    )
    _start(server)
    assert captured_payload["weights_capture"] == {
        "result_path": "/tmp/fake_weights.json",
        "cache_root": str(tmp_path / "split-cache"),
    }


def test_weights_capture_base_falls_back_to_root(captured_payload, tmp_path: Path):
    server = RootstockServer(
        env_name="uma",
        checkpoint="uma-s-1p1",
        device="cpu",
        root=tmp_path,
        weights_capture_path="/tmp/fake_weights.json",
    )
    _start(server)
    assert captured_payload["weights_capture"]["cache_root"] == str(tmp_path)
