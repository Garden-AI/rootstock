# Rootstock v0.5 Design: Pre-HPC Polish

## Overview

Final polish before HPC deployment. Two targeted fixes:

1. **Proper Python version parsing** — Replace fragile string manipulation with `packaging` library
2. **Sanity check benchmark** — Reproducible MD simulation to verify IPC overhead remains acceptable

---

## 1. Python Version Parsing Fix

### Problem

Current code in `cli.py`:

```python
python_version = requires_python.replace(">=", "")
```

This breaks on common PEP 440 specifiers:

| Input | Current Output | Expected |
|-------|----------------|----------|
| `>=3.10` | `3.10` | `3.10` ✓ |
| `>=3.10,<3.13` | `3.10,<3.13` | `3.10` ✗ |
| `~=3.10` | `~=3.10` | `3.10` ✗ |
| `>=3.10.0` | `3.10.0` | `3.10` ✓ |

### Solution

Use the `packaging` library (already a transitive dependency via pip/uv) to properly parse version specifiers:

```python
def extract_minimum_python_version(requires_python: str) -> str:
    """
    Extract minimum Python version from a requires-python specifier.
    
    Handles PEP 440 version specifiers like:
        ">=3.10"        -> "3.10"
        ">=3.10,<3.13"  -> "3.10"
        "~=3.10"        -> "3.10"
        ">=3.10.0"      -> "3.10"  (normalized for uv)
    
    Args:
        requires_python: PEP 440 version specifier string
        
    Returns:
        Minimum version string suitable for `uv venv --python X.Y`
        
    Raises:
        ValueError: If no minimum version can be determined
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    
    spec_set = SpecifierSet(requires_python)
    
    min_version = None
    
    for spec in spec_set:
        # Operators that establish a lower bound
        if spec.operator in (">=", "~=", "=="):
            version = Version(spec.version)
            if min_version is None or version < min_version:
                min_version = version
        elif spec.operator == ">":
            # Strict greater-than: we can't determine exact minimum
            # but the version given is a reasonable approximation for uv
            version = Version(spec.version)
            if min_version is None or version < min_version:
                min_version = version
    
    if min_version is None:
        raise ValueError(
            f"Cannot determine minimum Python version from '{requires_python}'. "
            "Specifier must include >=, ~=, ==, or > constraint."
        )
    
    # Return major.minor only (uv expects "3.10" not "3.10.0")
    return f"{min_version.major}.{min_version.minor}"
```

### Updated `cmd_build`

```python
def cmd_build(args) -> int:
    # ... existing code ...
    
    # Parse PEP 723 metadata
    content = env_source.read_text()
    metadata = parse_pep723_metadata(content)
    
    dependencies = metadata.get("dependencies", [])
    requires_python = metadata.get("requires-python", ">=3.10")
    
    # Extract minimum version properly
    try:
        python_version = extract_minimum_python_version(requires_python)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    print(f"  Python: {requires_python} -> {python_version}")
    # ... rest of build ...
```

### Add `packaging` as explicit dependency

Update `pyproject.toml`:

```toml
dependencies = [
    "ase>=3.22",
    "numpy>=1.24",
    "packaging>=21.0",  # For version specifier parsing
    "tomli>=2.0; python_version < '3.11'",
]
```

Note: `packaging` is already installed transitively (via pip, setuptools, etc.), but making it explicit documents the dependency.

---

## 2. Sanity Check Benchmark

### Purpose

Before deploying to HPC, we need a reproducible benchmark that:

1. Runs a realistic workload (not microbenchmarks)
2. Measures IPC overhead after calculator initialization
3. Provides clear pass/fail criteria
4. Can be run on Modal (for CI) and HPC (for validation)

### Success Criteria

- **Target**: <5% IPC overhead for 1000+ atom systems
- **Acceptable**: <10% overhead (flag for investigation)
- **Fail**: >10% overhead (block deployment)

### Benchmark Design

**Workload**: 5000-step MD simulation using ASE's VelocityVerlet integrator

This is what we tested in v0.2 and achieved ~2.9% overhead. It represents:
- A realistic use case (MD is the primary application)
- Enough steps to amortize any per-calculation variance
- ~5 minutes runtime (long enough to be meaningful, short enough for CI)

### Implementation

New file: `benchmarks/sanity_check.py`

```python
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

import numpy as np
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

Direct MACE:        {self.direct_time_s:.1f}s
RootstockCalculator: {self.rootstock_time_s:.1f}s
Overhead:           {self.overhead_s:.1f}s ({self.overhead_pct:.1f}%)

Status: {self.status}
"""


def run_md_direct(
    atoms,
    n_steps: int,
    model: str = "medium",
    device: str = "cuda",
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
) -> float:
    """
    Run MD simulation with direct MACE calculator.
    
    Returns wall-clock time for the MD loop only (excludes setup).
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
```

### Modal Integration

Add to `modal_app.py`:

```python
@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",
    timeout=1800,  # 30 min max
)
def sanity_check(
    n_atoms: int = 1000,
    n_steps: int = 5000,
) -> dict:
    """
    Run sanity check benchmark on Modal.
    
    This is the gate check before HPC deployment.
    """
    import sys
    sys.path.insert(0, "/root")
    
    from benchmarks.sanity_check import sanity_check as run_sanity_check
    
    result = run_sanity_check(
        n_atoms=n_atoms,
        n_steps=n_steps,
        model="medium",
        device="cuda",
        cluster="modal",
    )
    
    return {
        "system_size": result.system_size,
        "n_steps": result.n_steps,
        "direct_time_s": result.direct_time_s,
        "rootstock_time_s": result.rootstock_time_s,
        "overhead_pct": result.overhead_pct,
        "status": result.status,
    }
```

### Expected Results

Based on v0.2 benchmarking (~1000 atoms, 5000 steps):

| Metric | Expected | Acceptable |
|--------|----------|------------|
| Direct MACE | ~280s | - |
| Rootstock | ~290s | <310s |
| Overhead | ~2.9% | <5% |

---

## 3. Implementation Checklist

- [ ] Add `extract_minimum_python_version()` to `rootstock/cli.py`
- [ ] Update `cmd_build()` to use new parser
- [ ] Add `packaging>=21.0` to `pyproject.toml` dependencies
- [ ] Create `benchmarks/sanity_check.py`
- [ ] Add `sanity_check` function to `modal_app.py`
- [ ] Run sanity check on Modal before HPC deployment
- [ ] Document expected results in CLAUDE.md

---

## 4. Deployment Gate

Before deploying to Della (or any HPC):

```bash
# 1. Run sanity check on Modal
modal run modal_app.py::sanity_check

# 2. Verify result
# - Status should be PASS or MARGINAL
# - Overhead should be <5% (PASS) or <10% (MARGINAL)

# 3. If FAIL, investigate before proceeding
```

This gives us confidence that the v0.4 pre-built environment changes haven't introduced any performance regressions before we deploy to production HPC systems.