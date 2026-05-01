import pytest

from rootstock import worker


class FakeWorker:
    def __init__(self, socket_name, calculator, log=None, socket_path=None):
        self.calculator = calculator

    def run(self):
        return None


def test_run_worker_uses_setup_default_when_model_is_empty(monkeypatch):
    monkeypatch.setattr(worker, "MLIPWorker", FakeWorker)
    calls = []

    def setup(model="default-checkpoint", device="cuda"):
        calls.append((model, device))
        return {"model": model, "device": device}

    worker.run_worker(
        setup_fn=setup,
        model="",
        device="cpu",
        socket_path="/tmp/rootstock-test.sock",
    )

    assert calls == [("default-checkpoint", "cpu")]


def test_run_worker_rejects_missing_required_model(monkeypatch):
    monkeypatch.setattr(worker, "MLIPWorker", FakeWorker)

    def setup(model, device="cuda"):
        return {"model": model, "device": device}

    with pytest.raises(ValueError, match="requires one"):
        worker.run_worker(
            setup_fn=setup,
            model="",
            device="cpu",
            socket_path="/tmp/rootstock-test.sock",
        )
