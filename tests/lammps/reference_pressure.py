"""
Generate reference structure and pressure via ASE + RootstockCalculator.

Produces:
  - reference_structure.data  (LAMMPS data file)
  - Prints ASE pressure in bar for comparison with LAMMPS

Usage:
  python reference_pressure.py

Requires rootstock to be installed and the mace environment built.
"""

import numpy as np
from ase.build import bulk
from ase.io import write as ase_write

from rootstock.calculator import RootstockCalculator


def main():
    # 2x2x2 Cu FCC supercell (cubic) with random perturbation
    rng = np.random.default_rng(42)
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 2, 2)
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)

    # Calculate stress with rootstock
    with RootstockCalculator(
        cluster="della",
        checkpoint="mace-mp-0-medium",
        device="cpu",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=True)  # eV/A^3, Voigt order

    print(f"Energy: {energy:.10f} eV")
    print(f"Max force component: {np.abs(forces).max():.10f} eV/A")

    # Compute pressure
    # stress is in eV/A^3 (Voigt: xx, yy, zz, yz, xz, xy)
    # pressure = -1/3 * trace(stress)
    volume = atoms.get_volume()
    pressure_eV_A3 = -np.mean(stress[:3])
    pressure_GPa = pressure_eV_A3 * 160.21766208
    pressure_bar = pressure_GPa * 1e4
    print(f"Volume: {volume:.6f} A^3")
    print(f"ASE pressure: {pressure_bar:.2f} bar")

    # Write LAMMPS data file
    ase_write("reference_structure.data", atoms, format="lammps-data")
    print("Wrote reference_structure.data")

    # Write reference forces
    with open("reference_forces.dat", "w") as f:
        f.write(f"# energy = {energy:.15e} eV\n")
        f.write(f"# pressure = {pressure_bar:.6f} bar\n")
        f.write(f"# natoms = {len(atoms)}\n")
        for i, (fx, fy, fz) in enumerate(forces, start=1):
            f.write(f"{i} {fx:.15e} {fy:.15e} {fz:.15e}\n")
    print("Wrote reference_forces.dat")


if __name__ == "__main__":
    main()
