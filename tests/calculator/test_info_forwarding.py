"""Client-side atoms.info handling: JSON-safe extraction and cache invalidation.

ASE's compare_atoms ignores atoms.info, so RootstockCalculator overrides
check_state — an info-only change (charge, spin, external_field) must reach
the worker, not be served from the cached result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from rootstock.calculator import RootstockCalculator, _json_safe_info

_MACE_ENV_SOURCE = '''\
"""MACE env."""

CHECKPOINTS = {
    "mace-mp-0-medium": "medium",
}


def setup(checkpoint, device="cuda"):
    return None
'''


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A fake install root with one env (mace) declaring mace-mp-0-medium."""
    env_dir = tmp_path / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_MACE_ENV_SOURCE)
    return tmp_path


def _calc_with_state(fake_root: Path, atoms: Atoms) -> RootstockCalculator:
    """A calculator that behaves as if it already calculated `atoms`."""
    calc = RootstockCalculator(checkpoint="mace-mp-0-medium", root=fake_root)
    calc.atoms = atoms.copy()
    return calc


def _h2(**info) -> Atoms:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.info.update(info)
    return atoms


class TestJsonSafeInfo:
    def test_plain_values_pass_through(self):
        info = {"charge": 1, "spin": 2, "name": "x", "flag": True, "temp": 0.5, "none": None}
        assert _json_safe_info(info) == info

    def test_numpy_scalar_converted(self):
        safe = _json_safe_info({"charge": np.int64(1), "temp": np.float64(0.5)})
        assert safe == {"charge": 1, "temp": 0.5}
        assert type(safe["charge"]) is int
        assert type(safe["temp"]) is float

    def test_numpy_array_converted_to_list(self):
        safe = _json_safe_info({"external_field": np.array([0.0, 0.0, 0.5])})
        assert safe == {"external_field": [0.0, 0.0, 0.5]}

    def test_unserializable_value_dropped(self):
        safe = _json_safe_info({"charge": 1, "obj": object()})
        assert safe == {"charge": 1}

    def test_non_string_key_dropped(self):
        safe = _json_safe_info({("a", "b"): 1, "charge": 1})
        assert safe == {"charge": 1}


class TestCheckStateInfo:
    def test_info_change_invalidates(self, fake_root: Path):
        calc = _calc_with_state(fake_root, _h2(charge=0))
        assert calc.check_state(_h2(charge=1)) == ["info"]

    def test_added_info_key_invalidates(self, fake_root: Path):
        calc = _calc_with_state(fake_root, _h2())
        assert calc.check_state(_h2(charge=1)) == ["info"]

    def test_identical_info_is_cached(self, fake_root: Path):
        calc = _calc_with_state(fake_root, _h2(charge=1))
        assert calc.check_state(_h2(charge=1)) == []

    def test_equal_array_values_are_cached(self, fake_root: Path):
        calc = _calc_with_state(fake_root, _h2(external_field=np.array([0.0, 0.0, 0.5])))
        assert calc.check_state(_h2(external_field=np.array([0.0, 0.0, 0.5]))) == []

    def test_unserializable_values_do_not_invalidate(self, fake_root: Path):
        # A value that can't cross the socket can't change the result either.
        calc = _calc_with_state(fake_root, _h2(obj=object()))
        assert calc.check_state(_h2(obj=object())) == []

    def test_geometry_change_still_reported(self, fake_root: Path):
        calc = _calc_with_state(fake_root, _h2(charge=0))
        moved = _h2(charge=1)
        moved.positions[1, 2] = 1.0
        assert "positions" in calc.check_state(moved)
