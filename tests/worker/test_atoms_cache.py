"""Tests for MLIPWorker._create_atoms cache invalidation.

The server re-sends INIT (numbers + pbc) every force cycle; the worker's
cached Atoms must be rebuilt whenever those differ from what it holds, not
only when the atom count changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from rootstock.worker import MLIPWorker


def _make_worker(numbers: list[int] | None, pbc: list[bool] | None = None) -> MLIPWorker:
    worker = MLIPWorker(calculator=None, socket_path="/tmp/unused")
    worker._atomic_numbers = numbers
    worker._pbc = pbc
    return worker


def _positions(n: int) -> np.ndarray:
    return np.arange(n * 3, dtype=float).reshape(n, 3)


CELL = np.eye(3) * 10.0


def test_first_call_builds_atoms_from_init_data():
    worker = _make_worker([1, 8], pbc=[True, False, True])
    atoms = worker._create_atoms(_positions(2), CELL)
    assert list(atoms.numbers) == [1, 8]
    assert list(atoms.pbc) == [True, False, True]


def test_unchanged_system_reuses_atoms_in_place():
    worker = _make_worker([1, 1])
    atoms1 = worker._create_atoms(_positions(2), CELL)
    new_positions = _positions(2) + 1.0
    atoms2 = worker._create_atoms(new_positions, CELL * 2)
    assert atoms2 is atoms1  # fast path: same object, updated in place
    np.testing.assert_array_equal(atoms2.positions, new_positions)
    np.testing.assert_array_equal(atoms2.cell[:], CELL * 2)


def test_same_count_composition_change_rebuilds():
    worker = _make_worker([1, 1])
    worker._create_atoms(_positions(2), CELL)

    worker._atomic_numbers = [1, 8]  # same count, different species
    atoms = worker._create_atoms(_positions(2), CELL)
    assert list(atoms.numbers) == [1, 8]


def test_pbc_change_rebuilds():
    worker = _make_worker([1, 1], pbc=[True, True, True])
    worker._create_atoms(_positions(2), CELL)

    worker._pbc = [False, False, False]  # same species, different pbc
    atoms = worker._create_atoms(_positions(2), CELL)
    assert list(atoms.pbc) == [False, False, False]


def test_atom_count_change_rebuilds():
    worker = _make_worker([1, 1])
    worker._create_atoms(_positions(2), CELL)

    worker._atomic_numbers = [1, 1, 8]
    atoms = worker._create_atoms(_positions(3), CELL)
    assert list(atoms.numbers) == [1, 1, 8]


def test_missing_numbers_raises():
    worker = _make_worker(None)
    with pytest.raises(RuntimeError, match="No atomic numbers"):
        worker._create_atoms(_positions(2), CELL)
