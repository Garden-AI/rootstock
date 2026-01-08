"""
Modal app for running Rootstock IPC benchmarks on GPU.

Usage:
    modal run modal_app.py::benchmark_all
    modal run modal_app.py::benchmark_size --size large
    modal run modal_app.py::benchmark_ipc_overhead
"""

import modal

# -----------------------------------------------------------------------------
# Modal configuration
# -----------------------------------------------------------------------------

app = modal.App("rootstock-ipc-benchmark")

# Image with MACE and dependencies
mace_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "ase>=3.22",
        "mace-torch>=0.3",
    )
    # Copy our package into the image
    .copy_local_dir("rootstock", "/root/rootstock")
    .copy_local_dir("benchmarks", "/root/benchmarks")
    .env({"PYTHONPATH": "/root"})
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
    model: str = "mace-mp-0",
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
    model: str = "mace-mp-0",
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
