"""Tests for the bytecode pre-compilation step of ``rootstock install``."""

from __future__ import annotations

import sys
from pathlib import Path

from rootstock.commands.install import _precompile_environment


def _make_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "env"
    tree.mkdir()
    (tree / "good.py").write_text("x = 1\n")
    return tree


def test_precompile_writes_pyc_in_tree(tmp_path: Path, monkeypatch):
    """.pyc must land next to the sources (world-visible), even when the
    maintainer's shell has a per-user PYTHONPYCACHEPREFIX exported."""
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "elsewhere"))
    tree = _make_tree(tmp_path)

    _precompile_environment(Path(sys.executable), tree)

    assert list((tree / "__pycache__").glob("good.*.pyc"))
    assert not (tmp_path / "elsewhere").exists()


def test_precompile_warns_but_does_not_raise_on_bad_files(tmp_path: Path, capsys):
    tree = _make_tree(tmp_path)
    (tree / "bad.py").write_text("def broken(:\n")

    _precompile_environment(Path(sys.executable), tree)  # must not raise

    out = capsys.readouterr().out
    assert "Warning: some files did not byte-compile" in out
    assert "bad.py" in out
    # The good file still got compiled.
    assert list((tree / "__pycache__").glob("good.*.pyc"))
