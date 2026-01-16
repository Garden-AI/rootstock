#!/usr/bin/env python
"""
MD Comparison Benchmark: Direct MACE vs Rootstock

This runs identical NVT molecular dynamics simulations using:
1. Direct MACE (in-process, baseline)
2. Rootstock (MACE in subprocess via i-PI protocol)

The goal is to demonstrate that a realistic ~5-10 minute MD run has
negligible overhead when using Rootstock's isolated environment approach.

Usage:
    # Via Modal (recommended)
    modal run benchmarks/md_comparison.py::run_md_comparison
    
    # Locally (requires GPU + MACE)
    python benchmarks/md_comparison.py
    
    # Quick test (fewer steps)
    python benchmarks/md_comparison.py --steps 500
"""

import argparse
import sys
import time
from pathlib import Path


def make_cu_system(n_atoms: int = 1000, temperature_K: float = 300.0):
    """
    Create a Cu FCC supercell with initial velocities.

    Args:
        n_atoms: Target number of atoms
        temperature_K: Initial temperature for velocity distribution

    Returns:
        ASE Atoms object ready for MD
    """
    import numpy as np
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    
    # Cu FCC: 4 atoms per cubic unit cell
    n = int(round((n_atoms / 4) ** (1/3)))
    n = max(2, n)
    
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (n, n, n)
    
    # Add small random displacement to break symmetry
    rng = np.random.default_rng(42)
    atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    
    return atoms


def run_nvt_md(atoms, calc, n_steps: int, timestep_fs: float = 1.0,
               temperature_K: float = 300.0, log_interval: int = 100):
    """
    Run NVT molecular dynamics using Langevin thermostat.

    Args:
        atoms: ASE Atoms object
        calc: ASE calculator
        n_steps: Number of MD steps
        timestep_fs: Timestep in femtoseconds
        temperature_K: Target temperature
        log_interval: Print progress every N steps

    Returns:
        Dict with trajectory data and timing info
    """
    import numpy as np
    from ase import units
    from ase.md.langevin import Langevin
    
    atoms = atoms.copy()
    atoms.calc = calc
    
    # Langevin thermostat
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=0.01 / units.fs,  # Moderate friction
    )
    
    # Storage for trajectory data
    energies = []
    temperatures = []
    times_per_step = []
    
    def record():
        energies.append(atoms.get_potential_energy())
        temperatures.append(atoms.get_temperature())
    
    # Initial state
    record()
    
    # Run MD
    step = 0
    while step < n_steps:
        t0 = time.perf_counter()
        dyn.run(1)  # Single step for accurate timing
        times_per_step.append(time.perf_counter() - t0)
        
        step += 1
        
        if step % log_interval == 0:
            record()
            avg_time = np.mean(times_per_step[-log_interval:]) * 1000
            print(f"  Step {step:5d}/{n_steps}: "
                  f"E={energies[-1]:.3f} eV, "
                  f"T={temperatures[-1]:.1f} K, "
                  f"avg {avg_time:.1f} ms/step")
    
    return {
        "energies": energies,
        "temperatures": temperatures,
        "times_per_step_ms": [t * 1000 for t in times_per_step],
        "final_positions": atoms.positions.copy(),
    }


def run_comparison(
    n_atoms: int = 1000,
    n_steps: int = 5000,
    timestep_fs: float = 1.0,
    temperature_K: float = 300.0,
    model: str = "medium",
    device: str = "cuda",
):
    """
    Run MD comparison between direct MACE and Rootstock.

    Args:
        n_atoms: Number of atoms in the system
        n_steps: Number of MD steps
        timestep_fs: Timestep in femtoseconds
        temperature_K: Temperature in Kelvin
        model: MACE model name
        device: Device (cuda or cpu)

    Returns:
        Dict with comparison results
    """
    import numpy as np

    print("=" * 70)
    print("Rootstock IPC Overhead: MD Comparison Benchmark")
    print("=" * 70)
    print(f"System: ~{n_atoms} Cu atoms")
    print(f"MD: {n_steps} steps NVT @ {temperature_K} K, dt={timestep_fs} fs")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Create initial system (same for both runs)
    print("\nCreating system...")
    atoms = make_cu_system(n_atoms, temperature_K)
    print(f"Actual system size: {len(atoms)} atoms")
    
    # =========================================================================
    # Direct MACE (baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIRECT MACE (baseline)")
    print("=" * 70)
    
    from mace.calculators import mace_mp
    direct_calc = mace_mp(model=model, device=device, default_dtype="float32")
    
    print("Running MD...")
    t0_direct = time.time()
    direct_results = run_nvt_md(
        atoms, 
        direct_calc, 
        n_steps=n_steps,
        timestep_fs=timestep_fs,
        temperature_K=temperature_K,
    )
    time_direct = time.time() - t0_direct
    
    print(f"\nDirect MACE completed in {time_direct:.1f} seconds "
          f"({time_direct/60:.2f} minutes)")
    
    # =========================================================================
    # Rootstock (via subprocess)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ROOTSTOCK (MACE in subprocess via i-PI)")
    print("=" * 70)
    
    # Add parent dir to path for imports
    repo_root = Path(__file__).parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    from rootstock import RootstockCalculator
    
    print("Starting Rootstock calculator (spawning worker)...")
    
    t0_rootstock = time.time()
    with RootstockCalculator(model=model, device=device) as rootstock_calc:
        print("Worker ready. Running MD...")
        rootstock_results = run_nvt_md(
            atoms,
            rootstock_calc,
            n_steps=n_steps,
            timestep_fs=timestep_fs,
            temperature_K=temperature_K,
        )
    time_rootstock = time.time() - t0_rootstock
    
    print(f"\nRootstock completed in {time_rootstock:.1f} seconds "
          f"({time_rootstock/60:.2f} minutes)")
    
    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    overhead_seconds = time_rootstock - time_direct
    overhead_pct = (overhead_seconds / time_direct) * 100
    
    # Per-step timing
    direct_mean_ms = np.mean(direct_results["times_per_step_ms"])
    rootstock_mean_ms = np.mean(rootstock_results["times_per_step_ms"])
    overhead_per_step_ms = rootstock_mean_ms - direct_mean_ms
    
    print("\nWall Clock Time:")
    print(f"  Direct:    {time_direct:7.1f} s  ({time_direct/60:.2f} min)")
    print(f"  Rootstock: {time_rootstock:7.1f} s  ({time_rootstock/60:.2f} min)")
    print(f"  Overhead:  {overhead_seconds:+7.1f} s  ({overhead_pct:+.1f}%)")
    
    print("\nPer-Step Timing:")
    print(f"  Direct:    {direct_mean_ms:.2f} ms/step")
    print(f"  Rootstock: {rootstock_mean_ms:.2f} ms/step")
    print(f"  Overhead:  {overhead_per_step_ms:+.2f} ms/step")
    
    # Validate results are physically similar
    direct_final_E = direct_results["energies"][-1]
    rootstock_final_E = rootstock_results["energies"][-1]
    direct_mean_T = np.mean(direct_results["temperatures"])
    rootstock_mean_T = np.mean(rootstock_results["temperatures"])
    
    print("\nPhysical Validation:")
    print(f"  Final energy - Direct:    {direct_final_E:.3f} eV")
    print(f"  Final energy - Rootstock: {rootstock_final_E:.3f} eV")
    print(f"  Mean temp - Direct:       {direct_mean_T:.1f} K")
    print(f"  Mean temp - Rootstock:    {rootstock_mean_T:.1f} K")
    
    # Note: Trajectories will diverge due to chaotic dynamics, but
    # energies and temperatures should be statistically similar
    
    print("\n" + "=" * 70)
    if overhead_pct < 5:
        verdict = "✓ SUCCESS"
        detail = "IPC overhead is acceptable (<5%)"
    elif overhead_pct < 10:
        verdict = "⚠ MARGINAL"
        detail = "IPC overhead is noticeable but may be acceptable (5-10%)"
    else:
        verdict = "✗ HIGH OVERHEAD"
        detail = f"IPC overhead exceeds 10% target ({overhead_pct:.1f}%)"
    
    print(f"{verdict}: {detail}")
    print("=" * 70)
    
    # One-liner for sharing
    print(f"\n📊 Summary: {len(atoms)} atoms, {n_steps} MD steps")
    print(f"   Direct: {time_direct/60:.1f} min → Rootstock: {time_rootstock/60:.1f} min "
          f"({overhead_pct:+.1f}% overhead)")
    
    # Return native Python types (not numpy) for Modal deserialization
    return {
        "n_atoms": int(len(atoms)),
        "n_steps": int(n_steps),
        "time_direct_s": float(time_direct),
        "time_rootstock_s": float(time_rootstock),
        "overhead_s": float(overhead_seconds),
        "overhead_pct": float(overhead_pct),
        "direct_ms_per_step": float(direct_mean_ms),
        "rootstock_ms_per_step": float(rootstock_mean_ms),
        "overhead_ms_per_step": float(overhead_per_step_ms),
    }


# =============================================================================
# Modal entry point
# =============================================================================

try:
    import modal
    
    app = modal.App("rootstock-md-comparison")
    
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.0",
            "numpy>=1.24",
            "ase>=3.22",
            "mace-torch>=0.3",
        )
        .env({"PYTHONPATH": "/root"})
        .add_local_dir("rootstock", "/root/rootstock")
        .add_local_dir("benchmarks", "/root/benchmarks")
    )
    
    @app.function(image=image, gpu="A10G", timeout=3600)
    def run_md_comparison(
        n_atoms: int = 1000,
        n_steps: int = 5000,
    ) -> dict:
        """Modal entry point for MD comparison."""
        import sys
        sys.path.insert(0, "/root")

        # Re-run the function defined above (now in scope via Modal serialization)
        return run_comparison(n_atoms=n_atoms, n_steps=n_steps)
    
    @app.local_entrypoint()
    def modal_main(n_atoms: int = 1000, n_steps: int = 5000):
        """Run MD comparison on Modal."""
        result = run_md_comparison.remote(n_atoms=n_atoms, n_steps=n_steps)
        print(f"\nModal result: {result}")

except ImportError:
    # Modal not installed, skip Modal-specific code
    pass


# =============================================================================
# Local entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare MD performance: Direct MACE vs Rootstock"
    )
    parser.add_argument(
        "--atoms", type=int, default=1000,
        help="Target number of atoms (default: 1000)"
    )
    parser.add_argument(
        "--steps", type=int, default=5000,
        help="Number of MD steps (default: 5000)"
    )
    parser.add_argument(
        "--timestep", type=float, default=1.0,
        help="Timestep in fs (default: 1.0)"
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0,
        help="Temperature in K (default: 300.0)"
    )
    parser.add_argument(
        "--model", default="medium",
        help="MACE model (default: medium)"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device (default: cuda)"
    )
    args = parser.parse_args()
    
    run_comparison(
        n_atoms=args.atoms,
        n_steps=args.steps,
        timestep_fs=args.timestep,
        temperature_K=args.temperature,
        model=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()