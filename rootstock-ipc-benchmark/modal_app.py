"""
Modal app for running Rootstock IPC benchmarks on GPU.

Usage:
    modal run modal_app.py::test_mace_environment
    modal run modal_app.py::test_chgnet_environment
    modal run modal_app.py::benchmark_v2
    modal run modal_app.py::benchmark_ipc_overhead
"""

import modal

# -----------------------------------------------------------------------------
# Modal configuration
# -----------------------------------------------------------------------------

app = modal.App("rootstock-ipc-benchmark")

# Image with uv for environment-based testing
# This image only needs rootstock + uv, as MACE/CHGNet are installed by uv run
rootstock_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")  # Needed for uv installer
    .pip_install(
        "torch>=2.0",  # Base torch for GPU support
        "numpy>=1.24",
        "ase>=3.22",
        "tomli>=2.0",  # Needed for PEP 723 parsing
    )
    # Install uv
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "PYTHONPATH": "/root",
        "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
    })
    # Copy our package (including pyproject.toml and README.md for proper package structure)
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
    .add_local_file("README.md", "/root/README.md")
    .add_local_dir("rootstock", "/root/rootstock")
    .add_local_dir("benchmarks", "/root/benchmarks")
    .add_local_dir("environments", "/root/environments")
)


# -----------------------------------------------------------------------------
# Environment Tests
# -----------------------------------------------------------------------------


@app.function(
    image=rootstock_image,
    gpu="A10G",
    timeout=1800,
)
def test_mace_environment(
    model: str = "medium",
    device: str = "cuda",
) -> dict:
    """
    Test the MACE environment file using RootstockCalculator.

    This validates the full workflow:
    1. Parse PEP 723 metadata from environments/mace_env.py
    2. Generate wrapper script
    3. Spawn worker via `uv run`
    4. Run calculations through i-PI protocol
    """
    import sys
    import time

    sys.path.insert(0, "/root")

    from ase.build import bulk

    from rootstock import RootstockCalculator
    from rootstock.pep723 import validate_environment_file

    env_path = "/root/environments/mace_env.py"

    print("=" * 60)
    print("Testing MACE Environment")
    print("=" * 60)

    # Validate environment file
    print("\n1. Validating environment file...")
    is_valid, msg = validate_environment_file(env_path)
    print(f"   Valid: {is_valid} ({msg})")
    if not is_valid:
        return {"success": False, "error": msg}

    # Create test system
    print("\n2. Creating test system...")
    atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)  # 500 atoms
    print(f"   Created: {len(atoms)} atoms")

    # Run calculation
    print(f"\n3. Running calculation (model={model}, device={device})...")
    start_time = time.time()

    try:
        with RootstockCalculator(
            environment=env_path,
            model=model,
            device=device,
            log=sys.stderr,
        ) as calc:
            atoms.calc = calc

            # Warmup
            print("   Warmup...")
            _ = atoms.get_potential_energy()

            # Timed calculations
            print("   Running 10 calculations...")
            times = []
            for i in range(10):
                t0 = time.time()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                times.append(time.time() - t0)

        total_time = time.time() - start_time

        import numpy as np

        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000

        print("\n4. Results:")
        print(f"   Energy: {energy:.6f} eV")
        print(f"   Forces shape: {forces.shape}")
        print(f"   Mean time: {mean_ms:.2f} ms")
        print(f"   Std time: {std_ms:.2f} ms")
        print(f"   Total time: {total_time:.1f} s")

        print("\n" + "=" * 60)
        print("MACE Environment Test: PASSED")
        print("=" * 60)

        return {
            "success": True,
            "environment": "mace",
            "model": model,
            "n_atoms": len(atoms),
            "energy": energy,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "total_time": total_time,
        }

    except Exception as e:
        import traceback

        print(f"\nError: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.function(
    image=rootstock_image,
    gpu="A10G",
    timeout=1800,
)
def test_chgnet_environment(
    device: str = "cuda",
) -> dict:
    """
    Test the CHGNet environment file using RootstockCalculator.
    """
    import sys
    import time

    sys.path.insert(0, "/root")

    from ase.build import bulk

    from rootstock import RootstockCalculator
    from rootstock.pep723 import validate_environment_file

    env_path = "/root/environments/chgnet_env.py"

    print("=" * 60)
    print("Testing CHGNet Environment")
    print("=" * 60)

    # Validate environment file
    print("\n1. Validating environment file...")
    is_valid, msg = validate_environment_file(env_path)
    print(f"   Valid: {is_valid} ({msg})")
    if not is_valid:
        return {"success": False, "error": msg}

    # Create test system
    print("\n2. Creating test system...")
    atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)  # 500 atoms
    print(f"   Created: {len(atoms)} atoms")

    # Run calculation (CHGNet uses default model when model is empty)
    print(f"\n3. Running calculation (device={device})...")
    start_time = time.time()

    try:
        with RootstockCalculator(
            environment=env_path,
            model="",  # CHGNet uses default model
            device=device,
            log=sys.stderr,
        ) as calc:
            atoms.calc = calc

            # Warmup
            print("   Warmup...")
            _ = atoms.get_potential_energy()

            # Timed calculations
            print("   Running 10 calculations...")
            times = []
            for i in range(10):
                t0 = time.time()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                times.append(time.time() - t0)

        total_time = time.time() - start_time

        import numpy as np

        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000

        print("\n4. Results:")
        print(f"   Energy: {energy:.6f} eV")
        print(f"   Forces shape: {forces.shape}")
        print(f"   Mean time: {mean_ms:.2f} ms")
        print(f"   Std time: {std_ms:.2f} ms")
        print(f"   Total time: {total_time:.1f} s")

        print("\n" + "=" * 60)
        print("CHGNet Environment Test: PASSED")
        print("=" * 60)

        return {
            "success": True,
            "environment": "chgnet",
            "n_atoms": len(atoms),
            "energy": energy,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "total_time": total_time,
        }

    except Exception as e:
        import traceback

        print(f"\nError: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------


@app.function(
    image=rootstock_image,
    gpu="A10G",
    timeout=3600,
)
def benchmark_v2(
    n_calls: int = 50,
    n_warmup: int = 5,
    model: str = "medium",
) -> dict:
    """
    Benchmark RootstockCalculator with MACE environment vs direct MACE.

    Compares:
    - Direct MACE calculation (baseline)
    - RootstockCalculator with MACE environment

    This validates that IPC overhead is acceptable (<5% for 1000+ atoms).
    """
    import sys
    import time

    sys.path.insert(0, "/root")

    import numpy as np

    from benchmarks.direct import benchmark_direct
    from benchmarks.systems import get_benchmark_system
    from rootstock import RootstockCalculator

    print("=" * 60)
    print("Rootstock IPC Benchmark")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Calls: {n_calls}, Warmup: {n_warmup}")

    results = {}

    for size in ["small", "medium", "large"]:
        atoms = get_benchmark_system(size)

        print(f"\n{'='*60}")
        print(f"System: {size} ({len(atoms)} atoms)")
        print("=" * 60)

        # Direct benchmark (baseline)
        print("\n[Direct MACE - baseline]")
        direct_result = benchmark_direct(
            atoms,
            n_calls=n_calls,
            n_warmup=n_warmup,
            model=model,
            device="cuda",
        )
        print(direct_result.summary())

        # Rootstock benchmark
        print("\n[RootstockCalculator - via uv run]")
        env_path = "/root/environments/mace_env.py"

        times = []
        with RootstockCalculator(
            environment=env_path,
            model=model,
            device="cuda",
        ) as calc:
            atoms.calc = calc

            # Warmup
            for _ in range(n_warmup):
                atoms.get_potential_energy()

            # Timed calls
            for _ in range(n_calls):
                t0 = time.time()
                atoms.get_potential_energy()
                times.append(time.time() - t0)

        times_ms = np.array(times) * 1000
        rootstock_mean_ms = np.mean(times_ms)
        rootstock_std_ms = np.std(times_ms)
        rootstock_p95_ms = np.percentile(times_ms, 95)

        print(f"  Mean: {rootstock_mean_ms:.2f} ms")
        print(f"  Std:  {rootstock_std_ms:.2f} ms")
        print(f"  P95:  {rootstock_p95_ms:.2f} ms")

        # Analysis
        overhead_ms = rootstock_mean_ms - direct_result.mean_ms
        overhead_pct = (overhead_ms / direct_result.mean_ms) * 100

        print("\n[Overhead Analysis]")
        print(f"  IPC overhead: {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")

        if overhead_pct < 5:
            status = "PASS (<5%)"
        elif overhead_pct < 10:
            status = "MARGINAL (5-10%)"
        else:
            status = "HIGH (>10%)"
        print(f"  Status: {status}")

        results[size] = {
            "n_atoms": len(atoms),
            "direct_mean_ms": direct_result.mean_ms,
            "rootstock_mean_ms": rootstock_mean_ms,
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct,
            "status": status,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - IPC Overhead")
    print("=" * 60)
    print(f"{'Size':<10} {'Atoms':<8} {'Direct':<12} {'Rootstock':<12} {'Overhead':<15}")
    print("-" * 60)
    for size, r in results.items():
        print(
            f"{size:<10} "
            f"{r['n_atoms']:<8} "
            f"{r['direct_mean_ms']:.2f} ms    "
            f"{r['rootstock_mean_ms']:.2f} ms    "
            f"{r['overhead_ms']:.2f} ms ({r['overhead_pct']:.1f}%)"
        )

    # Final verdict
    print("\n" + "=" * 60)
    large_result = results.get("large")
    if large_result:
        if large_result["overhead_pct"] < 5:
            print("SUCCESS: IPC overhead is acceptable for 1000+ atom systems")
        else:
            print("CONCERN: IPC overhead exceeds 5% target for 1000+ atoms")
    print("=" * 60)

    return results


@app.function(
    image=rootstock_image,
    gpu="A10G",
    timeout=600,
)
def benchmark_ipc_overhead(
    n_calls: int = 1000,
    n_warmup: int = 10,
) -> dict:
    """
    Measure pure IPC overhead without MLIP calculation.

    Uses a mock worker that returns zeros instantly.
    This isolates the socket communication overhead.
    """
    import sys
    sys.path.insert(0, "/root")

    from benchmarks.ipc import benchmark_ipc_overhead_only
    from benchmarks.systems import get_benchmark_system

    results = {}

    print("Measuring pure IPC overhead (no MLIP calculation)...")
    print(f"Calls: {n_calls}, Warmup: {n_warmup}")
    print()

    for size in ["small", "medium", "large", "xlarge"]:
        atoms = get_benchmark_system(size)

        print(f"{size} ({len(atoms)} atoms)...")
        result = benchmark_ipc_overhead_only(
            atoms,
            n_calls=n_calls,
            n_warmup=n_warmup,
        )

        print(f"  Mean: {result.mean_ms:.3f} ms")
        print(f"  Std:  {result.std_ms:.3f} ms")
        print(f"  P95:  {result.p95_ms:.3f} ms")

        results[size] = {
            "n_atoms": len(atoms),
            "mean_ms": result.mean_ms,
            "std_ms": result.std_ms,
            "p95_ms": result.p95_ms,
        }

    print("\nPure IPC overhead summary:")
    print("This is the theoretical minimum overhead from socket communication.")

    return results


# -----------------------------------------------------------------------------
# Local entry point
# -----------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Run the full benchmark suite."""
    results = benchmark_v2.remote()

    print("\n\nFinal Results (from Modal):")
    print("=" * 60)
    for size, r in results.items():
        print(f"{size}: {r['overhead_pct']:.1f}% overhead")
