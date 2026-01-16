"""
Modal app for running Rootstock IPC benchmarks on GPU.

Usage (v0.1 legacy benchmarks):
    modal run modal_app.py::benchmark_all
    modal run modal_app.py::benchmark_size --size large
    modal run modal_app.py::benchmark_ipc_overhead

Usage (v0.2 environment tests):
    modal run modal_app.py::test_mace_environment
    modal run modal_app.py::test_chgnet_environment
    modal run modal_app.py::benchmark_v2
"""

import modal

# -----------------------------------------------------------------------------
# Modal configuration
# -----------------------------------------------------------------------------

app = modal.App("rootstock-ipc-benchmark")

# Image with MACE and dependencies (v0.1 legacy)
mace_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "ase>=3.22",
        "mace-torch>=0.3",
    )
    .env({"PYTHONPATH": "/root"})
    # Copy our package into the image (must be last)
    .add_local_dir("rootstock", "/root/rootstock")
    .add_local_dir("benchmarks", "/root/benchmarks")
)

# Image with uv for v0.2 environment-based testing
# This image only needs rootstock + uv, as MACE/CHGNet are installed by uv run
rootstock_v2_image = (
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
# Benchmark functions
# -----------------------------------------------------------------------------

@app.function(
    image=mace_image,
    gpu="A10G",  # Good balance of performance and availability
    timeout=1800,  # 30 minutes
)
def benchmark_size(
    size: str = "medium",
    n_calls: int = 50,
    n_warmup: int = 5,
    model: str = "medium",
) -> dict:
    """
    Run benchmark for a single system size.
    
    Args:
        size: One of "small", "medium", "large", "xlarge"
        n_calls: Number of timed calculations
        n_warmup: Number of warmup calculations
        model: MACE model name
    
    Returns:
        Dict with benchmark results
    """
    import sys
    sys.path.insert(0, "/root")
    
    from benchmarks.systems import get_benchmark_system
    from benchmarks.direct import benchmark_direct
    from benchmarks.ipc import benchmark_ipc
    
    atoms = get_benchmark_system(size)
    
    print(f"Benchmarking {size} system ({len(atoms)} atoms)...")
    print(f"Model: {model}, Calls: {n_calls}, Warmup: {n_warmup}")
    
    # Direct benchmark
    print("\nRunning direct benchmark...")
    direct_result = benchmark_direct(
        atoms,
        n_calls=n_calls,
        n_warmup=n_warmup,
        model=model,
        device="cuda",
    )
    print(direct_result.summary())
    
    # IPC benchmark
    print("\nRunning IPC benchmark...")
    ipc_result = benchmark_ipc(
        atoms,
        n_calls=n_calls,
        n_warmup=n_warmup,
        model=model,
        device="cuda",
    )
    print(ipc_result.summary())
    
    # Analysis
    overhead_ms = ipc_result.mean_ms - direct_result.mean_ms
    overhead_pct = (overhead_ms / direct_result.mean_ms) * 100
    
    print(f"\nOverhead: {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")
    
    return {
        "size": size,
        "n_atoms": len(atoms),
        "n_calls": n_calls,
        "direct": {
            "mean_ms": direct_result.mean_ms,
            "std_ms": direct_result.std_ms,
            "p95_ms": direct_result.p95_ms,
        },
        "ipc": {
            "mean_ms": ipc_result.mean_ms,
            "std_ms": ipc_result.std_ms,
            "p95_ms": ipc_result.p95_ms,
        },
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
    }


@app.function(
    image=mace_image,
    gpu="A10G",
    timeout=3600,  # 1 hour for full benchmark
)
def benchmark_all(
    n_calls: int = 50,
    n_warmup: int = 5,
    model: str = "medium",
) -> list[dict]:
    """
    Run benchmarks for all system sizes.
    
    Args:
        n_calls: Number of timed calculations per size
        n_warmup: Number of warmup calculations
        model: MACE model name
        
    Returns:
        List of result dicts for each size
    """
    import sys
    sys.path.insert(0, "/root")
    
    from benchmarks.systems import list_benchmark_systems, get_benchmark_system
    from benchmarks.direct import benchmark_direct
    from benchmarks.ipc import benchmark_ipc
    
    sizes = list(list_benchmark_systems().keys())
    results = []
    
    print("=" * 60)
    print("Rootstock IPC Overhead Benchmark")
    print(f"Model: {model}")
    print(f"Calls per size: {n_calls}")
    print("=" * 60)
    
    for size in sizes:
        atoms = get_benchmark_system(size)
        
        print(f"\n{'='*60}")
        print(f"System: {size} ({len(atoms)} atoms)")
        print("=" * 60)
        
        # Direct benchmark
        print("\n[Direct - baseline]")
        direct_result = benchmark_direct(
            atoms,
            n_calls=n_calls,
            n_warmup=n_warmup,
            model=model,
            device="cuda",
        )
        print(direct_result.summary())
        
        # IPC benchmark
        print("\n[IPC - via subprocess]")
        ipc_result = benchmark_ipc(
            atoms,
            n_calls=n_calls,
            n_warmup=n_warmup,
            model=model,
            device="cuda",
        )
        print(ipc_result.summary())
        
        # Analysis
        overhead_ms = ipc_result.mean_ms - direct_result.mean_ms
        overhead_pct = (overhead_ms / direct_result.mean_ms) * 100
        
        print(f"\n[Overhead Analysis]")
        print(f"  IPC overhead: {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")
        
        if overhead_pct < 5:
            status = "✓ PASS (<5%)"
        elif overhead_pct < 10:
            status = "⚠ MARGINAL (5-10%)"
        else:
            status = "✗ HIGH OVERHEAD (>10%)"
        print(f"  Status: {status}")
        
        results.append({
            "size": size,
            "n_atoms": len(atoms),
            "direct_mean_ms": direct_result.mean_ms,
            "direct_std_ms": direct_result.std_ms,
            "ipc_mean_ms": ipc_result.mean_ms,
            "ipc_std_ms": ipc_result.std_ms,
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct,
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Size':<10} {'Atoms':<8} {'Direct':<12} {'IPC':<12} {'Overhead':<15}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['size']:<10} "
            f"{r['n_atoms']:<8} "
            f"{r['direct_mean_ms']:.2f} ms    "
            f"{r['ipc_mean_ms']:.2f} ms    "
            f"{r['overhead_ms']:.2f} ms ({r['overhead_pct']:.1f}%)"
        )
    
    # Final verdict
    print("\n" + "=" * 60)
    large_result = next((r for r in results if r["size"] == "large"), None)
    if large_result:
        if large_result["overhead_pct"] < 5:
            print("✓ SUCCESS: IPC overhead is acceptable for 1000+ atom systems")
        else:
            print("✗ CONCERN: IPC overhead exceeds 5% target for 1000+ atoms")
    print("=" * 60)
    
    return results


@app.function(
    image=mace_image,
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
    
    from benchmarks.systems import get_benchmark_system
    from benchmarks.ipc import benchmark_ipc_overhead_only
    
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
# v0.2 Environment Tests
# -----------------------------------------------------------------------------


@app.function(
    image=rootstock_v2_image,
    gpu="A10G",
    timeout=1800,
)
def test_mace_environment(
    model: str = "medium",
    device: str = "cuda",
) -> dict:
    """
    Test the MACE environment file using the v0.2 RootstockCalculator.

    This validates the full v0.2 workflow:
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
    print("Testing MACE Environment (v0.2)")
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

        print(f"\n4. Results:")
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
    image=rootstock_v2_image,
    gpu="A10G",
    timeout=1800,
)
def test_chgnet_environment(
    device: str = "cuda",
) -> dict:
    """
    Test the CHGNet environment file using the v0.2 RootstockCalculator.
    """
    import sys
    import time

    sys.path.insert(0, "/root")

    from ase.build import bulk

    from rootstock import RootstockCalculator
    from rootstock.pep723 import validate_environment_file

    env_path = "/root/environments/chgnet_env.py"

    print("=" * 60)
    print("Testing CHGNet Environment (v0.2)")
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

    # Run calculation (CHGNet uses default model when model=None)
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

        print(f"\n4. Results:")
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


@app.function(
    image=rootstock_v2_image,
    gpu="A10G",
    timeout=3600,
)
def benchmark_v2(
    n_calls: int = 50,
    n_warmup: int = 5,
    model: str = "medium",
) -> dict:
    """
    Benchmark v0.2 with MACE environment vs direct MACE.

    Compares:
    - Direct MACE calculation (baseline)
    - v0.2 RootstockCalculator with MACE environment

    This validates that v0.2 overhead is still acceptable (<5% for 1000+ atoms).
    """
    import sys
    import time

    sys.path.insert(0, "/root")

    import numpy as np
    from ase.build import bulk

    from benchmarks.direct import benchmark_direct
    from benchmarks.systems import get_benchmark_system
    from rootstock import RootstockCalculator

    print("=" * 60)
    print("Rootstock v0.2 Benchmark")
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

        # v0.2 benchmark
        print("\n[v0.2 RootstockCalculator - via uv run]")
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
        v2_mean_ms = np.mean(times_ms)
        v2_std_ms = np.std(times_ms)
        v2_p95_ms = np.percentile(times_ms, 95)

        print(f"  Mean: {v2_mean_ms:.2f} ms")
        print(f"  Std:  {v2_std_ms:.2f} ms")
        print(f"  P95:  {v2_p95_ms:.2f} ms")

        # Analysis
        overhead_ms = v2_mean_ms - direct_result.mean_ms
        overhead_pct = (overhead_ms / direct_result.mean_ms) * 100

        print(f"\n[Overhead Analysis]")
        print(f"  v0.2 overhead: {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")

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
            "v2_mean_ms": v2_mean_ms,
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct,
            "status": status,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - v0.2 Overhead")
    print("=" * 60)
    print(f"{'Size':<10} {'Atoms':<8} {'Direct':<12} {'v0.2':<12} {'Overhead':<15}")
    print("-" * 60)
    for size, r in results.items():
        print(
            f"{size:<10} "
            f"{r['n_atoms']:<8} "
            f"{r['direct_mean_ms']:.2f} ms    "
            f"{r['v2_mean_ms']:.2f} ms    "
            f"{r['overhead_ms']:.2f} ms ({r['overhead_pct']:.1f}%)"
        )

    # Final verdict
    print("\n" + "=" * 60)
    large_result = results.get("large")
    if large_result:
        if large_result["overhead_pct"] < 5:
            print("SUCCESS: v0.2 overhead is acceptable for 1000+ atom systems")
        else:
            print("CONCERN: v0.2 overhead exceeds 5% target for 1000+ atoms")
    print("=" * 60)

    return results


# -----------------------------------------------------------------------------
# Local entry point
# -----------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Run the full benchmark suite."""
    results = benchmark_all.remote()

    print("\n\nFinal Results (from Modal):")
    print("=" * 60)
    for r in results:
        print(f"{r['size']}: {r['overhead_pct']:.1f}% overhead")
