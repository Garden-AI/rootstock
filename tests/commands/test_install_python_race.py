"""Publishing downloaded interpreters into the shared ``.python/`` dir.

Each build downloads its interpreter to a private tempdir, then publishes it
into ``{root}/.python/``. The old code copied straight to the destination
behind a check-then-act ``exists()`` guard, so two parallel builds wanting the
same interpreter collided: the loser's ``copytree`` raised, and a concurrent
reader could see a half-copied tree. Publication now stages under a unique
name inside ``.python/`` and atomically renames into place.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rootstock.operations import _publish_python_interpreter

INTERP = "cpython-3.11.9-linux-x86_64-gnu"


def _make_download(tmp_path: Path, name: str = "download") -> Path:
    """A fake `uv python install` result: one interpreter dir + one loose file."""
    tmp_python_dir = tmp_path / name / ".python"
    interp = tmp_python_dir / INTERP
    (interp / "bin").mkdir(parents=True)
    (interp / "bin" / "python").write_text("#!fake\n")
    (tmp_python_dir / ".lock").write_text("")
    return tmp_python_dir


def test_fresh_publish_lands_in_place(tmp_path):
    download = _make_download(tmp_path)
    install_dir = tmp_path / ".python"
    install_dir.mkdir()

    _publish_python_interpreter(download, install_dir, None)

    assert (install_dir / INTERP / "bin" / "python").exists()
    assert (install_dir / ".lock").is_file()
    assert not list(install_dir.glob("*.installing.*"))


def test_existing_interpreter_is_left_alone(tmp_path):
    download = _make_download(tmp_path)
    install_dir = tmp_path / ".python"
    existing = install_dir / INTERP
    existing.mkdir(parents=True)
    (existing / "already-here").touch()

    _publish_python_interpreter(download, install_dir, None)

    assert (existing / "already-here").exists()
    assert not (existing / "bin").exists()


def test_missing_download_dir_is_a_noop(tmp_path):
    install_dir = tmp_path / ".python"
    install_dir.mkdir()

    _publish_python_interpreter(tmp_path / "nope" / ".python", install_dir, None)

    assert list(install_dir.iterdir()) == []


def test_losing_the_rename_race_is_tolerated(tmp_path, monkeypatch):
    """Another build renames a complete copy in while ours is staging: our
    rename fails, but the winner's copy is as good as ours — no error, and
    our staging copy is swept."""
    download = _make_download(tmp_path)
    install_dir = tmp_path / ".python"
    install_dir.mkdir()

    real_copytree = shutil.copytree

    def racing_copytree(src, dst, *args, **kwargs):
        # copytree recurses through shutil.copytree for subdirectories; only
        # the top-level interpreter copy simulates the racing winner.
        result = real_copytree(src, dst, *args, **kwargs)
        if Path(src).name == INTERP:
            winner = install_dir / INTERP
            if not winner.exists():
                winner.mkdir()
                (winner / "winner-marker").touch()
        return result

    monkeypatch.setattr("rootstock.operations.shutil.copytree", racing_copytree)

    _publish_python_interpreter(download, install_dir, None)

    assert (install_dir / INTERP / "winner-marker").exists()
    assert not list(install_dir.glob("*.installing.*"))


def test_staging_deleted_underfoot_is_tolerated(tmp_path, monkeypatch):
    """A winner's sweep may delete a live loser's staging mid-copy: the
    loser's copytree fails, but the destination exists, so it carries on."""
    download = _make_download(tmp_path)
    install_dir = tmp_path / ".python"
    install_dir.mkdir()

    def swept_copytree(src, dst, **kwargs):
        winner = install_dir / Path(src).name
        winner.mkdir(parents=True, exist_ok=True)
        (winner / "winner-marker").touch()
        raise FileNotFoundError(f"{dst} deleted by a concurrent sweep")

    monkeypatch.setattr("rootstock.operations.shutil.copytree", swept_copytree)

    _publish_python_interpreter(download, install_dir, None)

    assert (install_dir / INTERP / "winner-marker").exists()


def test_real_failure_still_raises(tmp_path, monkeypatch):
    """OSErrors are only swallowed when another build actually won."""
    download = _make_download(tmp_path)
    install_dir = tmp_path / ".python"
    install_dir.mkdir()

    def broken_copytree(src, dst, **kwargs):
        raise PermissionError("disk on fire")

    monkeypatch.setattr("rootstock.operations.shutil.copytree", broken_copytree)

    with pytest.raises(PermissionError, match="disk on fire"):
        _publish_python_interpreter(download, install_dir, None)


def test_crashed_build_leftovers_are_swept(tmp_path):
    """Staging dirs abandoned by a killed build are reclaimed once the
    interpreter is in place."""
    download = _make_download(tmp_path)
    install_dir = tmp_path / ".python"
    stale = install_dir / f"{INTERP}.installing.99999"
    stale.mkdir(parents=True)
    (stale / "half-copied").touch()
    (install_dir / ".lock.installing.99999").touch()

    _publish_python_interpreter(download, install_dir, None)

    assert (install_dir / INTERP / "bin" / "python").exists()
    assert not stale.exists()
    assert not list(install_dir.glob("*.installing.*"))


def test_concurrent_publishes_of_same_interpreter(tmp_path):
    """N builds publishing the same interpreter at once: all succeed, one
    complete copy lands, no staging garbage remains."""
    install_dir = tmp_path / ".python"
    install_dir.mkdir()
    downloads = [_make_download(tmp_path, name=f"download{i}") for i in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(_publish_python_interpreter, d, install_dir, None) for d in downloads
        ]
        for future in futures:
            future.result()  # raises if any publish blew up

    assert (install_dir / INTERP / "bin" / "python").exists()
    assert not list(install_dir.glob("*.installing.*"))
