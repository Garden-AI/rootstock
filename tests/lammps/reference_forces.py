"""
Generate reference structure and forces via ASE + RootstockCalculator.

Produces:
  - reference_structure.data  (LAMMPS data file)
  - reference_forces.dat      (atom_id fx fy fz)

Usage:
  python reference_forces.py

Requires rootstock to be installed and the mace environment built.
"""

import numpy as np
from ase.build import bulk
from ase.io import write as ase_write

from rootstock.calculator import RootstockCalculator


def main():
    # 2x2x2 Cu FCC supercell with random perturbation
    atoms = bulk("Cu", "fcc", a=3.615) * (2, 2, 2)
    rng = np.random.default_rng(42)
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)

    # Calculate forces with rootstock
    with RootstockCalculator(
        cluster="della",
        checkpoint="mace-mp-0-medium",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()

    print(f"Energy: {energy:.10f} eV")
    print(f"Max force component: {np.abs(forces).max():.10f} eV/A")

    # Write LAMMPS data file
    ase_write("reference_structure.data", atoms, format="lammps-data")
    print("Wrote reference_structure.data")

    # Write reference forces
    with open("reference_forces.dat", "w") as f:
        f.write(f"# energy = {energy:.15e} eV\n")
        f.write(f"# natoms = {len(atoms)}\n")
        for i, (fx, fy, fz) in enumerate(forces, start=1):
            f.write(f"{i} {fx:.15e} {fy:.15e} {fz:.15e}\n")
    print("Wrote reference_forces.dat")


if __name__ == "__main__":
    main()
