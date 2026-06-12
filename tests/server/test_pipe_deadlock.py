"""Regression test for the worker stdout/stderr pipe deadlock.

`RootstockServer` spawns the worker with ``stdout=PIPE, stderr=PIPE`` (server.py,
when ``log`` is None — the default) and never drains those pipes. The worker
loads the model in ``setup()`` *before* it connects to the socket
(``worker.run_worker``), so a model whose load is chatty enough to exceed the
~64 KB OS pipe buffer blocks on the write and never connects. The server then
loops forever in ``_accept_connection`` (it only watches ``proc.poll()``, which
stays None because the worker is alive — just blocked).

This test drives the *real* ``RootstockCalculator`` against a faked env whose
``setup()`` floods stderr before returning a trivial Lennard-Jones calculator.
No GPU, no real MLIP, no ``uv`` build: we symlink the env's ``bin/python`` to
the interpreter running the tests (which can already import rootstock + ase).

It is marked ``xfail(strict=True)``: it documents the open bug today, and the
moment the pipe handling is fixed it will XPASS and fail strict-xfail, which is
the signal to delete this marker and keep the test as a plain guard.
"""

from __future__ import annotations

import os
import signal
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from rootstock.calculator import RootstockCalculator

# Comfortably above the 64 KB pipe buffer on Linux/macOS, on stderr alone.
_FLOOD_BYTES = 256 * 1024

# How long we wait before declaring a deadlock. The healthy path (spawn python,
# import ase/numpy, flood, connect, run one LJ step) is ~1-3s; under the bug we
# wait this whole budget out, so keep it tight but not flaky.
_DEADLINE_S = 8.0

_CHECKPOINT = "deadlock-dummy"

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"),
    reason="deadline guard needs SIGALRM (POSIX only)",
)


class _Deadline(Exception):
    """Raised when the SIGALRM deadline fires — i.e. we deadlocked."""


@contextmanager
def deadline(seconds: float):
    """Raise ``_Deadline`` if the wrapped block runs longer than ``seconds``.

    Uses SIGALRM so it interrupts a blocking ``accept()`` in the server's
    connect loop, which a thread-join timeout could not do cleanly.
    """

    def _fire(signum, frame):
        raise _Deadline(f"no progress after {seconds}s — worker never connected")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def chatty_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A faked rootstock install whose env floods stderr during setup().

    Layout mirrors a real install enough for the server to spawn a worker:
        {root}/envs/chatty/bin/python      -> symlink to this interpreter
        {root}/envs/chatty/env_source.py   -> CHECKPOINTS + flooding setup()

    A real env has rootstock + the MLIP deps pip-installed into its own
    site-packages. We skip the (slow) venv build by symlinking ``bin/python``
    to the test interpreter and handing the worker a ``PYTHONPATH`` so it can
    still ``import rootstock`` / ``ase`` — CPython resolves a symlinked
    interpreter's prefix from the *symlink* location, which has no
    site-packages, so the import would otherwise fail. The worker inherits this
    env via ``os.environ.copy()`` in EnvironmentManager.get_environment_variables.
    """
    import ase

    import rootstock

    pythonpath = os.pathsep.join(
        sorted({str(Path(m.__file__).resolve().parents[1]) for m in (ase, rootstock)})
    )
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    env_dir = tmp_path / "envs" / "chatty"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)

    (env_dir / "env_source.py").write_text(
        "CHECKPOINTS = {%r: 'dummy'}\n\n"
        "def setup(checkpoint, device='cpu', **kwargs):\n"
        "    import sys\n"
        "    # Simulate a noisy model load (torch/HF/mace warnings) that exceeds\n"
        "    # the OS pipe buffer *before* the worker connects to the socket.\n"
        "    sys.stderr.write('X' * %d)\n"
        "    sys.stderr.flush()\n"
        "    from ase.calculators.lj import LennardJones\n"
        "    return LennardJones()\n" % (_CHECKPOINT, _FLOOD_BYTES)
    )
    return tmp_path


@pytest.mark.xfail(
    strict=True,
    reason="worker stdout/stderr pipe is never drained; chatty setup() deadlocks "
    "the server's accept loop. See GH issue (fill in #). Remove this marker "
    "once the worker output handling is fixed.",
)
def test_chatty_worker_setup_does_not_deadlock(chatty_env_root: Path):
    from ase.build import molecule

    atoms = molecule("H2O")  # non-periodic: avoids the stress path entirely

    calc = RootstockCalculator(
        checkpoint=_CHECKPOINT,
        root=chatty_env_root,
        device="cpu",
    )
    try:
        atoms.calc = calc
        with deadline(_DEADLINE_S):
            energy = atoms.get_potential_energy()
        # If we got here the worker connected despite flooding stderr.
        assert energy == pytest.approx(atoms.calc.results["energy"])
    finally:
        calc.close()  # terminates the (possibly pipe-blocked) worker
