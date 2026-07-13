"""Worker startup knobs via ROOTSTOCK_* environment variables.

Workers are frozen inside built envs; env vars set at spawn are the one
config channel a newer client can always reach an old worker through.
"""

from __future__ import annotations

import io

import pytest

from rootstock import worker as worker_module
from rootstock.worker import MLIPWorker, run_worker


class _ConnectRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, socket_path, **kwargs):
        self.calls.append((socket_path, kwargs))
        raise SystemExit(0)  # stop before any protocol traffic


@pytest.fixture
def recorded_connect(monkeypatch):
    recorder = _ConnectRecorder()
    monkeypatch.setattr(worker_module, "connect_unix_socket", recorder)
    return recorder


def _connect(recorder) -> dict:
    worker = MLIPWorker(socket_name="test", calculator=None)
    with pytest.raises(SystemExit):
        worker._connect()
    assert len(recorder.calls) == 1
    return recorder.calls[0][1]


def test_connect_defaults(recorded_connect):
    kwargs = _connect(recorded_connect)
    assert kwargs["max_retries"] == 50
    assert kwargs["retry_delay"] == 0.1


def test_connect_env_overrides(recorded_connect, monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRIES", "300")
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY", "0.5")
    kwargs = _connect(recorded_connect)
    assert kwargs["max_retries"] == 300
    assert kwargs["retry_delay"] == 0.5


def test_invalid_env_values_fall_back_to_defaults(recorded_connect, monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRIES", "banana")
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY", "")
    kwargs = _connect(recorded_connect)
    assert kwargs["max_retries"] == 50
    assert kwargs["retry_delay"] == 0.1


def test_invalid_env_value_is_logged(recorded_connect, monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRIES", "banana")
    log = io.StringIO()
    worker = MLIPWorker(socket_name="test", calculator=None, log=log)
    with pytest.raises(SystemExit):
        worker._connect()
    assert "ROOTSTOCK_WORKER_CONNECT_RETRIES" in log.getvalue()


def _bail_out_setup(checkpoint, device, **kwargs):
    raise SystemExit(0)


def test_worker_log_env_opens_file(tmp_path, monkeypatch):
    log_path = tmp_path / "worker.log"
    monkeypatch.setenv("ROOTSTOCK_WORKER_LOG", str(log_path))
    with pytest.raises(SystemExit):
        run_worker(
            setup_fn=_bail_out_setup,
            checkpoint="mace-mp-0-medium",
            device="cpu",
            socket_path="/tmp/does_not_matter",
        )
    assert "[Worker] Calling setup" in log_path.read_text()


def test_worker_log_env_ignored_when_client_attached_a_log(tmp_path, monkeypatch):
    log_path = tmp_path / "worker.log"
    monkeypatch.setenv("ROOTSTOCK_WORKER_LOG", str(log_path))
    client_log = io.StringIO()
    with pytest.raises(SystemExit):
        run_worker(
            setup_fn=_bail_out_setup,
            checkpoint="mace-mp-0-medium",
            device="cpu",
            socket_path="/tmp/does_not_matter",
            log=client_log,
        )
    assert not log_path.exists()
    assert "[Worker] Calling setup" in client_log.getvalue()


def test_unopenable_log_path_is_ignored(monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_WORKER_LOG", "/nonexistent-dir/worker.log")
    with pytest.raises(SystemExit):
        run_worker(
            setup_fn=_bail_out_setup,
            checkpoint="mace-mp-0-medium",
            device="cpu",
            socket_path="/tmp/does_not_matter",
        )
