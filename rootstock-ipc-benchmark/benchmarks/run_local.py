#!/usr/bin/env python
"""
Run benchmarks locally (requires GPU with MACE installed).

Usage:
    python -m benchmarks.run_local
    python -m benchmarks.run_local --sizes small medium
    python -m benchmarks.run_local --ipc-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def run_all_benchmarks(
    sizes: list[str] = None,
    n_calls: int = 50,
    n_warmup: int = 5,
    model: str = "medium",
    device: str = "cuda",
    output_file: str = None,
    ipc_overhead_only: bool = False,
):
    """
    Run all benchmarks and print results.
    
    Args:
        sizes: List of system sizes to benchmark
        n_calls: Number of timed calls per benchmark
        n_warmup: Number of warmup calls
        model: MACE model name
        device: Device to run on
        output_file: Optional JSON file to save results
        ipc_overhead_only: If True, only measure IPC overhead (no MLIP)
    """
    from .systems import get_benchmark_system, list_benchmark_systems
    from .direct import benchmark_direct, BenchmarkResult
    from .ipc import benchmark_ipc, benchmark_ipc_overhead_only
    
    if sizes is None:
        sizes = list(list_benchmark_systems().keys())
    
    results = []
    
    print("=" * 60)
    print("Rootstock IPC Overhead Benchmark")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Calls per benchmark: {n_calls}")
    print("=" * 60)
    print()
    
    if ipc_overhead_only:
        # Only measure IPC overhead with mock worker
        print("Measuring IPC overhead only (no MLIP)...")
        for size in sizes:
            atoms = get_benchmark_system(size)
            print(f"\n--- {size}: {len(atoms)} atoms ---")
            
            result = benchmark_ipc_overhead_only(
                atoms,
                n_calls=n_calls * 10,  # More calls since it's fast
                n_warmup=n_warmup,
            )
            print(result.summary())
            results.append({
                "benchmark": result.name,
                "size": size,
                "n_atoms": result.n_atoms,
                "n_calls": result.n_calls,
                "mean_ms": result.mean_ms,
                "std_ms": result.std_ms,
                "p95_ms": result.p95_ms,
            })
    else:
        # Full benchmark: direct vs IPC
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
                device=device,
            )
            print(direct_result.summary())
            
            # IPC benchmark
            print("\n[IPC - via subprocess]")
            ipc_result = benchmark_ipc(
                atoms,
                n_calls=n_calls,
                n_warmup=n_warmup,
                model=model,
                device=device,
            )
            print(ipc_result.summary())
            
            # Overhead analysis
            overhead_ms = ipc_result.mean_ms - direct_result.mean_ms
            overhead_pct = (overhead_ms / direct_result.mean_ms) * 100
            
            print(f"\n[Overhead Analysis]")
            print(f"  IPC overhead: {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")
            
            if overhead_pct < 5:
                status = "✓ PASS"
            elif overhead_pct < 10:
                status = "⚠ MARGINAL"
            else:
                status = "✗ HIGH OVERHEAD"
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
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if not ipc_overhead_only:
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
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "device": device,
                "n_calls": n_calls,
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Rootstock IPC benchmarks locally")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["small", "medium", "large", "xlarge"],
        help="System sizes to benchmark (default: all)",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=50,
        help="Number of timed calls per benchmark",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        help="Number of warmup calls",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="MACE model name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--ipc-only",
        action="store_true",
        help="Only measure IPC overhead (no MLIP calculation)",
    )
    
    args = parser.parse_args()
    
    run_all_benchmarks(
        sizes=args.sizes,
        n_calls=args.n_calls,
        n_warmup=args.n_warmup,
        model=args.model,
        device=args.device,
        output_file=args.output,
        ipc_overhead_only=args.ipc_only,
    )


if __name__ == "__main__":
    main()
