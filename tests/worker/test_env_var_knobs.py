"""Worker startup knobs via ROOTSTOCK_* environment variables.

Workers are frozen inside built envs; env vars set at spawn are the one
config channel a newer client can always reach an old worker through. They
are read once at startup into the frozen WorkerConfig singleton.
"""

from __future__ import annotations

import dataclasses
import io

import pytest

from rootstock import worker as worker_module
from rootstock.worker import MLIPWorker, run_worker
from rootstock.worker_config import (
    DEFAULT_CONNECT_RETRIES,
    DEFAULT_CONNECT_RETRY_DELAY,
    WorkerConfig,
    get_worker_config,
)


@pytest.fixture(autouse=True)
def _fresh_config_singleton():
    """Isolate the process-wide config cache between tests."""
    get_worker_config.cache_clear()
    yield
    get_worker_config.cache_clear()


# --- WorkerConfig unit tests --------------------------------------------------


def test_defaults_from_empty_environ():
    config = WorkerConfig.from_env({})
    assert config.connect_retries == DEFAULT_CONNECT_RETRIES
    assert config.connect_retry_delay == DEFAULT_CONNECT_RETRY_DELAY
    assert config.log_target is None
    assert config.warnings == ()


def test_env_overrides():
    config = WorkerConfig.from_env(
        {
            "ROOTSTOCK_WORKER_CONNECT_RETRIES": "300",
            "ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY": "0.5",
            "ROOTSTOCK_WORKER_LOG": "stderr",
        }
    )
    assert config.connect_retries == 300
    assert config.connect_retry_delay == 0.5
    assert config.log_target == "stderr"
    assert config.warnings == ()


def test_invalid_values_fall_back_with_warnings():
    config = WorkerConfig.from_env(
        {
            "ROOTSTOCK_WORKER_CONNECT_RETRIES": "banana",
            "ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY": "",
        }
    )
    assert config.connect_retries == DEFAULT_CONNECT_RETRIES
    assert config.connect_retry_delay == DEFAULT_CONNECT_RETRY_DELAY
    assert len(config.warnings) == 2
    assert any("ROOTSTOCK_WORKER_CONNECT_RETRIES" in w for w in config.warnings)
    assert any("ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY" in w for w in config.warnings)


def test_config_is_frozen():
    config = WorkerConfig.from_env({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.connect_retries = 1


def test_singleton_reads_process_environ_once(monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRIES", "7")
    first = get_worker_config()
    assert first.connect_retries == 7
    # Later env changes don't leak into an already-started worker
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRIES", "8")
    assert get_worker_config() is first


def test_open_log_resolves_file_path(tmp_path):
    log_path = tmp_path / "worker.log"
    config = WorkerConfig.from_env({"ROOTSTOCK_WORKER_LOG": str(log_path)})
    log = config.open_log()
    try:
        print("hello", file=log)
    finally:
        log.close()
    assert "hello" in log_path.read_text()


def test_open_log_tolerates_unopenable_path():
    config = WorkerConfig.from_env({"ROOTSTOCK_WORKER_LOG": "/nonexistent-dir/worker.log"})
    assert config.open_log() is None


# --- Integration: MLIPWorker._connect -----------------------------------------


class _ConnectRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, socket_path, **kwargs):
        self.calls.append((socket_path, kwargs))
        raise SystemExit(0)  # stop before any protocol traffic


def test_connect_uses_config_values(monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRIES", "300")
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY", "0.5")
    recorder = _ConnectRecorder()
    monkeypatch.setattr(worker_module, "connect_unix_socket", recorder)

    worker = MLIPWorker(socket_name="test", calculator=None)
    with pytest.raises(SystemExit):
        worker._connect()

    (_, kwargs) = recorder.calls[0]
    assert kwargs["max_retries"] == 300
    assert kwargs["retry_delay"] == 0.5


# --- Integration: run_worker startup -------------------------------------------


def _bail_out_setup(checkpoint, device, **kwargs):
    raise SystemExit(0)


def _run_worker_briefly(**kwargs):
    with pytest.raises(SystemExit):
        run_worker(
            setup_fn=_bail_out_setup,
            checkpoint="mace-mp-0-medium",
            device="cpu",
            socket_path="/tmp/does_not_matter",
            **kwargs,
        )


def test_worker_log_env_opens_file(tmp_path, monkeypatch):
    log_path = tmp_path / "worker.log"
    monkeypatch.setenv("ROOTSTOCK_WORKER_LOG", str(log_path))
    _run_worker_briefly()
    assert "[Worker] Calling setup" in log_path.read_text()


def test_worker_log_env_ignored_when_client_attached_a_log(tmp_path, monkeypatch):
    log_path = tmp_path / "worker.log"
    monkeypatch.setenv("ROOTSTOCK_WORKER_LOG", str(log_path))
    client_log = io.StringIO()
    _run_worker_briefly(log=client_log)
    assert not log_path.exists()
    assert "[Worker] Calling setup" in client_log.getvalue()


def test_config_warnings_logged_once_at_startup(monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_WORKER_CONNECT_RETRIES", "banana")
    client_log = io.StringIO()
    _run_worker_briefly(log=client_log)
    assert "ROOTSTOCK_WORKER_CONNECT_RETRIES" in client_log.getvalue()
