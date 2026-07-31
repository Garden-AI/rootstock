"""Tests for MLIPWorker._create_atoms cache invalidation.

The server re-sends INIT (numbers + pbc) every force cycle; the worker's
cached Atoms must be rebuilt whenever those differ from what it holds, not
only when the atom count changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from rootstock.worker import MLIPWorker


class _RecordingCalculator:
    """Stands in for the inner ASE calculator; records reset() calls."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def _make_worker(
    numbers: list[int] | None,
    pbc: list[bool] | None = None,
    info: dict | None = None,
    calculator=None,
) -> MLIPWorker:
    worker = MLIPWorker(calculator=calculator, socket_path="/tmp/unused")
    worker._atomic_numbers = numbers
    worker._pbc = pbc
    if info is not None:
        worker._info = info
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


def test_info_applied_on_build():
    worker = _make_worker([1, 8], info={"charge": 1, "spin": 2})
    atoms = worker._create_atoms(_positions(2), CELL)
    assert atoms.info["charge"] == 1
    assert atoms.info["spin"] == 2


def test_numeric_list_info_becomes_array():
    worker = _make_worker([1, 8], info={"external_field": [0.0, 0.0, 0.5], "label": ["a", "b"]})
    atoms = worker._create_atoms(_positions(2), CELL)
    field = atoms.info["external_field"]
    assert isinstance(field, np.ndarray)
    np.testing.assert_allclose(field, [0.0, 0.0, 0.5])
    assert atoms.info["label"] == ["a", "b"]  # non-numeric lists stay lists


def test_info_change_updates_atoms_and_resets_calculator():
    calc = _RecordingCalculator()
    worker = _make_worker([1, 1], info={"charge": 0}, calculator=calc)
    atoms1 = worker._create_atoms(_positions(2), CELL)
    assert calc.resets == 0

    worker._info = {"charge": 1}
    atoms2 = worker._create_atoms(_positions(2), CELL)
    assert atoms2 is atoms1  # info alone must not force a rebuild
    assert atoms2.info["charge"] == 1
    assert calc.resets == 1  # geometry is unchanged, so only reset() forces a recompute


def test_unchanged_info_does_not_reset():
    calc = _RecordingCalculator()
    worker = _make_worker([1, 1], info={"charge": 1}, calculator=calc)
    worker._create_atoms(_positions(2), CELL)

    worker._info = {"charge": 1}  # fresh dict, equal content
    worker._create_atoms(_positions(2) + 1.0, CELL)
    assert calc.resets == 0


def test_removed_info_key_is_dropped():
    calc = _RecordingCalculator()
    worker = _make_worker([1, 1], info={"charge": 1}, calculator=calc)
    worker._create_atoms(_positions(2), CELL)

    worker._info = {}
    atoms = worker._create_atoms(_positions(2), CELL)
    assert "charge" not in atoms.info
    assert calc.resets == 1


def test_rebuild_carries_new_info():
    worker = _make_worker([1, 1], info={"charge": 0})
    worker._create_atoms(_positions(2), CELL)

    worker._atomic_numbers = [1, 8]  # composition change forces a rebuild
    worker._info = {"charge": -1}
    atoms = worker._create_atoms(_positions(2), CELL)
    assert atoms.info["charge"] == -1
