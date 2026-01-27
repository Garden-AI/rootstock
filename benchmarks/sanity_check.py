"""
Sanity check benchmark for Rootstock.

Runs a short MD simulation comparing direct MACE vs RootstockCalculator.
This validates that IPC overhead remains acceptable before HPC deployment.

Usage:
    # On Modal
    modal run modal_app.py::sanity_check

    # Local (requires MACE environment)
    python -m benchmarks.sanity_check --direct-only
"""

import time
from dataclasses import dataclass

from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from benchmarks.systems import make_cu_supercell


@dataclass
class SanityCheckResult:
    """Results from sanity check benchmark."""

    system_size: int
    n_steps: int

    direct_time_s: float
    rootstock_time_s: float

    @property
    def overhead_s(self) -> float:
        return self.rootstock_time_s - self.direct_time_s

    @property
    def overhead_pct(self) -> float:
        return (self.overhead_s / self.direct_time_s) * 100

    @property
    def status(self) -> str:
        if self.overhead_pct < 5:
            return "PASS"
        elif self.overhead_pct < 10:
            return "MARGINAL"
        else:
            return "FAIL"

    def summary(self) -> str:
        return f"""
Rootstock Sanity Check
======================
System: {self.system_size} atoms, {self.n_steps} MD steps

Direct MACE:         {self.direct_time_s:.1f}s
RootstockCalculator: {self.rootstock_time_s:.1f}s
Overhead:            {self.overhead_s:.1f}s ({self.overhead_pct:.1f}%)

Status: {self.status}
"""


def run_md_direct(
    atoms,
    n_steps: int,
    model: str = "medium",
    device: str = "cuda",
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    mace_available: bool = True,
) -> float:
    """
    Run MD simulation with direct MACE calculator.

    Returns wall-clock time for the MD loop only (excludes setup).

    Args:
        mace_available: If False, tries to import mace and raises if not available.
    """
    from mace.calculators import mace_mp

    atoms = atoms.copy()

    # Setup calculator
    calc = mace_mp(model=model, device=device, default_dtype="float32")
    atoms.calc = calc

    # Initialize velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    # Create dynamics object
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    # Time the MD loop only
    t0 = time.perf_counter()
    dyn.run(n_steps)
    t1 = time.perf_counter()

    return t1 - t0


def run_md_rootstock(
    atoms,
    n_steps: int,
    cluster: str = "modal",
    model: str = "mace-medium",
    device: str = "cuda",
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    root: str | None = None,
) -> float:
    """
    Run MD simulation with RootstockCalculator.

    Returns wall-clock time for the MD loop only (excludes setup/teardown).
    """
    from rootstock import RootstockCalculator

    atoms = atoms.copy()

    # Setup calculator (outside timing)
    calc_kwargs = {"model": model, "device": device}
    if root:
        calc_kwargs["root"] = root
    else:
        calc_kwargs["cluster"] = cluster

    with RootstockCalculator(**calc_kwargs) as calc:
        atoms.calc = calc

        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

        # Create dynamics object
        dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

        # Time the MD loop only
        t0 = time.perf_counter()
        dyn.run(n_steps)
        t1 = time.perf_counter()

        return t1 - t0


def sanity_check(
    n_atoms: int = 1000,
    n_steps: int = 5000,
    model: str = "medium",
    device: str = "cuda",
    cluster: str = "modal",
    root: str | None = None,
) -> SanityCheckResult:
    """
    Run the sanity check benchmark.

    Args:
        n_atoms: Target number of atoms (actual may vary slightly)
        n_steps: Number of MD steps
        model: MACE model name (for direct) / model string (for rootstock)
        device: PyTorch device
        cluster: Cluster name for rootstock (ignored if root is set)
        root: Explicit root path for rootstock

    Returns:
        SanityCheckResult with timing data and pass/fail status
    """
    print(f"Creating {n_atoms}-atom Cu system...")
    atoms = make_cu_supercell(n_atoms)
    actual_atoms = len(atoms)
    print(f"  Actual size: {actual_atoms} atoms")

    print(f"\nRunning direct MACE ({n_steps} steps)...")
    direct_time = run_md_direct(
        atoms,
        n_steps=n_steps,
        model=model,
        device=device,
    )
    print(f"  Completed in {direct_time:.1f}s")

    print(f"\nRunning RootstockCalculator ({n_steps} steps)...")
    rootstock_time = run_md_rootstock(
        atoms,
        n_steps=n_steps,
        model=f"mace-{model}",
        device=device,
        cluster=cluster,
        root=root,
    )
    print(f"  Completed in {rootstock_time:.1f}s")

    result = SanityCheckResult(
        system_size=actual_atoms,
        n_steps=n_steps,
        direct_time_s=direct_time,
        rootstock_time_s=rootstock_time,
    )

    print(result.summary())

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rootstock sanity check benchmark")
    parser.add_argument("--atoms", type=int, default=1000, help="Target atom count")
    parser.add_argument("--steps", type=int, default=5000, help="MD steps")
    parser.add_argument("--model", default="medium", help="MACE model")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--cluster", default="modal", help="Cluster name")
    parser.add_argument("--root", help="Explicit root path")
    parser.add_argument("--direct-only", action="store_true", help="Only run direct benchmark")

    args = parser.parse_args()

    if args.direct_only:
        atoms = make_cu_supercell(args.atoms)
        t = run_md_direct(atoms, args.steps, args.model, args.device)
        print(f"Direct MACE: {t:.1f}s for {len(atoms)} atoms, {args.steps} steps")
    else:
        result = sanity_check(
            n_atoms=args.atoms,
            n_steps=args.steps,
            model=args.model,
            device=args.device,
            cluster=args.cluster,
            root=args.root,
        )

        # Exit with error code if failed
        if result.status == "FAIL":
            exit(1)
