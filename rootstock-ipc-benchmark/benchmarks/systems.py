"""
Test system generation for benchmarks.

Creates Cu supercells of various sizes for benchmarking.
Cu is chosen because:
1. Simple FCC structure
2. Well-supported by all MLIPs
3. Easy to generate large systems
"""

from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import bulk


def make_cu_supercell(
    target_atoms: int,
    rattling: float = 0.05,
    seed: int = 42,
) -> Atoms:
    """
    Create a Cu FCC supercell with approximately the target number of atoms.
    
    Args:
        target_atoms: Target number of atoms (actual may differ slightly)
        rattling: Random displacement magnitude in Angstrom
        seed: Random seed for reproducibility
        
    Returns:
        ASE Atoms object with Cu supercell
    """
    # Cu FCC has 4 atoms per conventional cell
    # For N atoms, we need N/4 unit cells
    # For a roughly cubic supercell, we want n³ * 4 ≈ target
    # So n ≈ (target/4)^(1/3)
    
    n = int(round((target_atoms / 4) ** (1/3)))
    n = max(1, n)  # At least 1x1x1
    
    # Create supercell
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (n, n, n)
    
    # Add random rattling to break symmetry
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0, rattling, atoms.positions.shape)
    
    return atoms


# Pre-defined test systems
BENCHMARK_SYSTEMS = {
    "small": 64,      # ~64 atoms, 4x4x4 primitive cells
    "medium": 256,    # ~256 atoms
    "large": 1000,    # ~1000 atoms
    "xlarge": 4000,   # ~4000 atoms
}


def get_benchmark_system(size: Literal["small", "medium", "large", "xlarge"]) -> Atoms:
    """
    Get a pre-defined benchmark system.
    
    Args:
        size: One of "small", "medium", "large", "xlarge"
        
    Returns:
        ASE Atoms object
    """
    target = BENCHMARK_SYSTEMS[size]
    return make_cu_supercell(target)


def list_benchmark_systems() -> dict[str, int]:
    """Return dict of benchmark system names to approximate atom counts."""
    return BENCHMARK_SYSTEMS.copy()
