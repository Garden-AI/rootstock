"""Client-side diagnostics use stdlib logging, not a hand-threaded log= file.

Server lifecycle goes to ``rootstock.server`` (INFO/DEBUG), the wire trace to
``rootstock.protocol`` (DEBUG). The worker side is deliberately untouched:
workers frozen inside built envs log to a file object controlled by the
ROOTSTOCK_WORKER_LOG env var (worker_config.py), so ``IPIProtocol`` still
honors an explicit ``log=`` file for that path.
"""

from __future__ import annotations

import io
import logging
import os
import socket
import sys
from pathlib import Path

import pytest
from ase.build import molecule

from rootstock.calculator import RootstockCalculator
from rootstock.protocol import IPIProtocol

_CHECKPOINT = "logging-dummy"


@pytest.fixture
def lj_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Faked install with a quiet LJ env (symlinked-interpreter harness)."""
    import ase

    import rootstock

    pythonpath = os.pathsep.join(
        sorted({str(Path(m.__file__).resolve().parents[1]) for m in (ase, rootstock)})
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    env_dir = tmp_path / "envs" / "lj"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(
        f"CHECKPOINTS = {{{_CHECKPOINT!r}: 'dummy'}}\n\n"
        "def setup(checkpoint, device='cpu', **kwargs):\n"
        "    from ase.calculators.lj import LennardJones\n"
        "    return LennardJones()\n"
    )
    return tmp_path


def test_lifecycle_and_wire_trace_reach_stdlib_logging(lj_env_root: Path, caplog):
    atoms = molecule("H2O")
    calc = RootstockCalculator(checkpoint=_CHECKPOINT, root=lj_env_root, device="cpu")
    try:
        atoms.calc = calc
        with caplog.at_level(logging.DEBUG, logger="rootstock"):
            atoms.get_potential_energy()
    finally:
        calc.close()

    server_records = [r for r in caplog.records if r.name == "rootstock.server"]
    protocol_records = [r for r in caplog.records if r.name == "rootstock.protocol"]

    assert any("Launched worker" in r.message for r in server_records)
    assert any("Worker connected" in r.message for r in server_records)
    assert any("send_posdata" in r.message for r in protocol_records)
    assert any("send_getforce" in r.message for r in protocol_records)


def test_quiet_at_default_level(lj_env_root: Path, caplog):
    """At WARNING (logging's default), a healthy run says nothing."""
    atoms = molecule("H2O")
    calc = RootstockCalculator(checkpoint=_CHECKPOINT, root=lj_env_root, device="cpu")
    try:
        atoms.calc = calc
        with caplog.at_level(logging.WARNING, logger="rootstock"):
            atoms.get_potential_energy()
    finally:
        calc.close()

    assert [r for r in caplog.records if r.name.startswith("rootstock")] == []


def test_calculator_log_kwarg_raises_with_guidance(tmp_path: Path):
    with pytest.raises(TypeError, match="stdlib logging"):
        RootstockCalculator(checkpoint="x", root=tmp_path, log=io.StringIO())


def test_protocol_still_honors_worker_side_log_file():
    """The worker path passes an explicit file object; that must keep working
    (worker logging is env-var-driven, not stdlib-logging-driven)."""
    a, b = socket.socketpair()
    try:
        log = io.StringIO()
        proto = IPIProtocol(a, log=log)
        proto.send_status()
        assert "send_status" in log.getvalue()
    finally:
        a.close()
        b.close()


def test_protocol_without_log_file_uses_stdlib_logging(caplog):
    a, b = socket.socketpair()
    try:
        proto = IPIProtocol(a)
        with caplog.at_level(logging.DEBUG, logger="rootstock.protocol"):
            proto.send_status()
        assert any("send_status" in r.message for r in caplog.records)
    finally:
        a.close()
        b.close()
