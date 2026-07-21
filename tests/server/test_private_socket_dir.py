"""Sockets live in a private 0700 directory, not world-visible /tmp.

Sockets used to follow the i-PI convention ``/tmp/ipi_<name>``: visible to
every local user, permissions at the mercy of the umask, and the path
squattable/race-able on shared login nodes. The server now creates each
socket inside a fresh ``mkdtemp`` directory (0700 by contract) and removes
the directory on stop.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from rootstock.protocol import create_private_socket_path
from rootstock.server import RootstockServer

_CHECKPOINT = "socket-dummy"


def test_socket_path_is_inside_fresh_private_dir():
    path = Path(create_private_socket_path("testsock"))
    try:
        parent = path.parent
        assert not path.exists()  # the path is reserved, not yet bound
        assert parent.is_dir()
        assert parent != Path(tempfile.gettempdir())  # never directly in /tmp
        mode = stat.S_IMODE(parent.stat().st_mode)
        assert mode == 0o700
        assert parent.stat().st_uid == os.getuid()
        assert path.name == "ipi_testsock"
    finally:
        os.rmdir(path.parent)


def test_socket_paths_are_unique_per_call():
    a = create_private_socket_path("same-name")
    b = create_private_socket_path("same-name")
    try:
        assert a != b
    finally:
        os.rmdir(Path(a).parent)
        os.rmdir(Path(b).parent)


def test_long_tempdir_falls_back_to_tmp(monkeypatch, tmp_path):
    """sun_path is ~104 bytes on macOS; a deep TMPDIR must not produce an
    unbindable socket path."""
    deep = tmp_path / ("d" * 80) / ("e" * 40)
    deep.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(deep))
    tempfile.tempdir = None  # force gettempdir() to re-read TMPDIR
    try:
        path = create_private_socket_path("fallback")
        assert len(path) < 100
        assert path.startswith("/tmp/")
        os.rmdir(Path(path).parent)
    finally:
        tempfile.tempdir = None


@pytest.fixture
def lj_env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A faked install with a quiet Lennard-Jones env (same trick as the
    pipe-deadlock test: symlink bin/python to this interpreter and hand the
    worker a PYTHONPATH so it can import rootstock + ase)."""
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


def test_server_socket_lives_in_private_dir_and_is_removed_on_stop(lj_env_root: Path):
    server = RootstockServer(
        env_name="lj",
        checkpoint=_CHECKPOINT,
        device="cpu",
        root=lj_env_root,
    )
    assert server.socket_path is None  # nothing on disk before start()

    server.start()
    try:
        socket_path = Path(server.socket_path)
        socket_dir = socket_path.parent
        assert socket_path.exists()
        assert socket_dir != Path("/tmp")
        assert stat.S_IMODE(socket_dir.stat().st_mode) == 0o700
    finally:
        server.stop()

    assert not socket_path.exists()
    assert not socket_dir.exists()
    assert server.socket_path is None
