"""Tests for verify_checkpoint's assertion logic.

End-to-end verification (real model, real worker) is exercised by the live
deployment validation called out in §9.2 of the design doc. These tests stub
out RootstockServer so we can drive each assertion branch deterministically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rootstock import verify


class _StubServer:
    """Stand-in for RootstockServer.

    Yields a fixed (energy, forces, virial) when calculate() is called.
    Records start/stop calls so we can confirm cleanup.
    """

    instances: list[_StubServer] = []

    def __init__(self, *, energy, forces, virial, raise_on_start=None, raise_on_calc=None, **_):
        self._energy = energy
        self._forces = forces
        self._virial = virial
        self._raise_on_start = raise_on_start
        self._raise_on_calc = raise_on_calc
        self.started = False
        self.stopped = False
        _StubServer.instances.append(self)

    def start(self):
        if self._raise_on_start is not None:
            raise self._raise_on_start
        self.started = True

    def calculate(self, *, positions, cell, atomic_numbers, pbc):
        if self._raise_on_calc is not None:
            raise self._raise_on_calc
        return self._energy, self._forces, self._virial

    def stop(self):
        self.stopped = True


@pytest.fixture
def stub_server(monkeypatch):
    """Patch RootstockServer in rootstock.verify to a stub with given outputs."""
    _StubServer.instances.clear()

    def _install(**outputs):
        def factory(**ctor_kwargs):
            # Forward configurable outputs from the closure, ignore real ctor args.
            return _StubServer(**outputs)

        monkeypatch.setattr("rootstock.server.RootstockServer", factory)
        return _StubServer

    return _install


def _ok_forces():
    # 3 atoms (H2O), non-zero non-symmetric forces.
    return np.array([[0.1, -0.2, 0.0], [-0.05, 0.1, 0.0], [-0.05, 0.1, 0.0]])


def _ok_virial():
    return np.eye(3) * 0.01


def test_verify_happy_path(stub_server):
    stub_server(energy=-10.5, forces=_ok_forces(), virial=_ok_virial())
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is True
    assert err is None


def test_verify_stops_server_even_on_success(stub_server):
    Stub = stub_server(energy=-10.5, forces=_ok_forces(), virial=_ok_virial())
    verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert all(s.stopped for s in Stub.instances)


def test_verify_rejects_nan_energy(stub_server):
    stub_server(energy=float("nan"), forces=_ok_forces(), virial=_ok_virial())
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "energy" in err


def test_verify_rejects_inf_energy(stub_server):
    stub_server(energy=float("inf"), forces=_ok_forces(), virial=_ok_virial())
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "energy" in err


def test_verify_rejects_wrong_force_shape(stub_server):
    bad = np.zeros((4, 3))  # 4 atoms reported for a 3-atom system
    bad[0, 0] = 1.0
    stub_server(energy=-1.0, forces=bad, virial=_ok_virial())
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "shape" in err


def test_verify_rejects_nonfinite_forces(stub_server):
    bad = _ok_forces().copy()
    bad[1, 1] = float("nan")
    stub_server(energy=-1.0, forces=bad, virial=_ok_virial())
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "forces" in err


def test_verify_rejects_all_zero_forces(stub_server):
    """The silent-failure guard — model returned zeros for everything."""
    stub_server(energy=-1.0, forces=np.zeros((3, 3)), virial=_ok_virial())
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "zero" in err.lower()


def test_verify_rejects_nonfinite_virial(stub_server):
    bad = _ok_virial().copy()
    bad[0, 0] = float("inf")
    stub_server(energy=-1.0, forces=_ok_forces(), virial=bad)
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "virial" in err


def test_verify_catches_server_start_failure(stub_server):
    stub_server(
        energy=0, forces=_ok_forces(), virial=_ok_virial(),
        raise_on_start=RuntimeError("worker died"),
    )
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "RuntimeError" in err
    assert "worker died" in err


def test_verify_catches_calculate_failure(stub_server):
    stub_server(
        energy=0, forces=_ok_forces(), virial=_ok_virial(),
        raise_on_calc=ValueError("CUDA out of memory"),
    )
    ok, err = verify.verify_checkpoint(Path("/tmp"), "mace", "mace-mp-0-medium", "cuda")
    assert ok is False
    assert "ValueError" in err


def test_smoke_test_atoms_has_charge_and_spin():
    atoms = verify._smoke_test_atoms()
    assert atoms.info["charge"] == 0
    assert atoms.info["spin"] == 1


def test_smoke_test_atoms_breaks_symmetry():
    atoms = verify._smoke_test_atoms()
    # The y-perturbation we apply means atom 1 is not exactly on the x-axis.
    assert abs(atoms.positions[1, 1]) > 1e-6
