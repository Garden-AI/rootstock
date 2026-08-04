"""``rootstock serve`` spools one usage record per worker session.

serve is the one entry point the RootstockServer hook can't cover — here the
i-PI server is external (e.g. the LAMMPS fix) and rootstock only runs the
worker — so cmd_serve records the session itself: client="serve",
n_calculations null (the parent process never sees the i-PI traffic).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

from rootstock.commands.serve import cmd_serve
from rootstock.environment import ResolvedCheckpoint
from rootstock.usage import usage_dir


class _FakeProc:
    def __init__(self, returncode):
        self.returncode = returncode

    def wait(self):
        return self.returncode

    def send_signal(self, signum):
        pass


def _wire_fake_worker(monkeypatch, returncode=0):
    @contextmanager
    def fake_spawn(root, env_name, wrapper, config, cache_root=None):
        yield SimpleNamespace(cmd=["worker"], env={}, cwd=None)

    monkeypatch.setattr("rootstock.spawn.spawn_in_env", fake_spawn)
    monkeypatch.setattr(
        "rootstock.environment.resolve_checkpoint",
        lambda root, ckpt, cluster=None: ResolvedCheckpoint(checkpoint=ckpt, env_name="mace"),
    )
    monkeypatch.setattr("rootstock.environment.get_env_python", lambda root, env: "python")
    monkeypatch.setattr(
        "rootstock.commands.serve.subprocess.Popen", lambda *a, **kw: _FakeProc(returncode)
    )
    # Don't rewire the test process's real signal handlers.
    monkeypatch.setattr("rootstock.commands.serve.signal.signal", lambda *a: None)


def _args(tmp_path):
    return SimpleNamespace(
        root=str(tmp_path),
        socket="/tmp/ipi.sock",
        checkpoint="mace-mp-0-medium",
        device="cuda",
        kwarg=None,
    )


def test_serve_spools_one_record(tmp_path, monkeypatch):
    usage_dir(tmp_path).mkdir()
    _wire_fake_worker(monkeypatch)

    assert cmd_serve(_args(tmp_path)) == 0

    (path,) = usage_dir(tmp_path).glob("*.json")
    record = json.loads(path.read_text())
    assert record["client"] == "serve"
    assert record["env"] == "mace"
    assert record["checkpoint"] == "mace-mp-0-medium"
    assert record["n_calculations"] is None
    assert record["duration_s"] >= 0


def test_serve_without_spool_records_nothing_and_returns_worker_rc(tmp_path, monkeypatch):
    _wire_fake_worker(monkeypatch, returncode=3)

    assert cmd_serve(_args(tmp_path)) == 3
    assert not usage_dir(tmp_path).exists()
