"""
Direct MACE benchmark (baseline).

Runs MACE in the same process without any IPC.
This establishes the baseline performance to compare against.
"""

import time
from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    name: str
    n_atoms: int
    n_calls: int
    times_ms: list[float]  # Per-call times in milliseconds
    
    @property
    def mean_ms(self) -> float:
        return np.mean(self.times_ms)
    
    @property
    def std_ms(self) -> float:
        return np.std(self.times_ms)
    
    @property
    def p95_ms(self) -> float:
        return np.percentile(self.times_ms, 95)
    
    @property
    def min_ms(self) -> float:
        return np.min(self.times_ms)
    
    @property
    def max_ms(self) -> float:
        return np.max(self.times_ms)
    
    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_atoms} atoms, {self.n_calls} calls\n"
            f"  mean: {self.mean_ms:.2f} ms\n"
            f"  std:  {self.std_ms:.2f} ms\n"
            f"  p95:  {self.p95_ms:.2f} ms\n"
            f"  range: [{self.min_ms:.2f}, {self.max_ms:.2f}] ms"
        )


def benchmark_direct(
    atoms: Atoms,
    n_calls: int = 100,
    n_warmup: int = 5,
    model: str = "medium",
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Benchmark MACE running directly in the same process.
    
    This is the baseline - no IPC overhead, everything in one process.
    
    Args:
        atoms: ASE Atoms object to use for benchmarking
        n_calls: Number of force calculations to time
        n_warmup: Number of warmup calls (not timed)
        model: MACE model name
        device: Device to run on
        
    Returns:
        BenchmarkResult with timing data
    """
    from mace.calculators import mace_mp
    
    # Load calculator
    calc = mace_mp(model=model, device=device, default_dtype="float32")
    atoms = atoms.copy()
    atoms.calc = calc
    
    # Warmup - ensures JIT compilation, GPU memory allocation, etc.
    for _ in range(n_warmup):
        atoms.get_forces()
        # Slightly perturb positions to avoid caching
        atoms.positions += np.random.normal(0, 0.001, atoms.positions.shape)
    
    # Timed runs
    times_ms = []
    for _ in range(n_calls):
        # Small perturbation to avoid any caching
        atoms.positions += np.random.normal(0, 0.001, atoms.positions.shape)
        
        t0 = time.perf_counter()
        atoms.get_forces()
        t1 = time.perf_counter()
        
        times_ms.append((t1 - t0) * 1000)
    
    return BenchmarkResult(
        name="direct",
        n_atoms=len(atoms),
        n_calls=n_calls,
        times_ms=times_ms,
    )
