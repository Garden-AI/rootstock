"""
Compare LAMMPS forces against ASE reference.

Reads:
  - reference_forces.dat   (from reference_forces.py)
  - lammps_forces.dump     (from test_forces.lammps)

Asserts max force difference < 1e-6 eV/A.

Usage:
  python compare_forces.py
"""

import sys

import numpy as np


def read_reference(path="reference_forces.dat"):
    """Read reference forces from our custom format."""
    forces = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            # atom_id fx fy fz
            forces.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(forces)


def read_lammps_dump(path="lammps_forces.dump"):
    """Read forces from LAMMPS custom dump file."""
    forces = {}
    in_atoms = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ITEM: ATOMS"):
                in_atoms = True
                continue
            if line.startswith("ITEM:"):
                in_atoms = False
                continue
            if in_atoms:
                parts = line.split()
                atom_id = int(parts[0])
                fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
                forces[atom_id] = [fx, fy, fz]

    # Sort by atom ID and return as array
    natoms = max(forces.keys())
    result = np.zeros((natoms, 3))
    for i in range(1, natoms + 1):
        result[i - 1] = forces[i]
    return result


def main():
    ref = read_reference()
    lmp = read_lammps_dump()

    if ref.shape != lmp.shape:
        print(f"FAIL: shape mismatch — reference {ref.shape}, LAMMPS {lmp.shape}")
        sys.exit(1)

    diff = np.abs(ref - lmp)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print("Force comparison:")
    print(f"  Atoms:          {len(ref)}")
    print(f"  Max difference: {max_diff:.2e} eV/A")
    print(f"  Mean difference: {mean_diff:.2e} eV/A")

    threshold = 1e-6
    if max_diff < threshold:
        print(f"  PASS (< {threshold} eV/A)")
    else:
        print(f"  FAIL (>= {threshold} eV/A)")
        # Show worst atoms
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Worst atom: {worst[0]+1}, component: {'xyz'[worst[1]]}")
        print(f"    Reference: {ref[worst[0]]}")
        print(f"    LAMMPS:    {lmp[worst[0]]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
