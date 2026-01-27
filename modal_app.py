"""
Modal app for running Rootstock v0.3 tests and benchmarks.

Usage:
    # Initialize the volume (first time only)
    modal run modal_app.py::init_rootstock_volume

    # Test new v0.3 API with caching
    modal run modal_app.py::test_new_api

    # Inspect cache contents
    modal run modal_app.py::inspect_cache

    # Legacy tests (still work)
    modal run modal_app.py::test_mace_environment
    modal run modal_app.py::benchmark_v2
"""

import modal

# -----------------------------------------------------------------------------
# Modal configuration
# -----------------------------------------------------------------------------

app = modal.App("rootstock-v03")

# Persistent volume for rootstock root directory (environments + caches)
rootstock_volume = modal.Volume.from_name("rootstock-data", create_if_missing=True)

# Base image with uv
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "ase>=3.22",
        "tomli>=2.0",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "PYTHONPATH": "/root",
        "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
    })
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
    .add_local_file("README.md", "/root/README.md")
    .add_local_dir("rootstock", "/root/rootstock")
    .add_local_dir("benchmarks", "/root/benchmarks")
    .add_local_dir("environments", "/root/environments")
)


# -----------------------------------------------------------------------------
# Volume Initialization
# -----------------------------------------------------------------------------


@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    timeout=600,
)
def init_rootstock_volume():
    """One-time setup: create directory structure and copy environment files."""
    import shutil
    from pathlib import Path

    root = Path("/vol/rootstock")

    # Clean up old files that might conflict (can't rmtree the mount point itself)
    for subdir in ["environments", "cache"]:
        path = root / subdir
        if path.exists():
            shutil.rmtree(path)

    # Create directory structure
    # Note: We only cache HuggingFace model weights on the volume.
    # uv cache uses local storage because Modal volumes don't support lock files.
    (root / "environments").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "huggingface").mkdir(parents=True, exist_ok=True)

    # Define environment files
    mace_env = '''# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///
"""MACE environment for Rootstock."""


def setup(model: str, device: str = "cuda"):
    """Load a MACE calculator."""
    from mace.calculators import mace_mp

    return mace_mp(model=model, device=device, default_dtype="float32")
'''

    chgnet_env = '''# /// script
# requires-python = ">=3.10"
# dependencies = ["chgnet>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///
"""CHGNet environment for Rootstock."""


def setup(model: str = "", device: str = "cuda"):
    """Load a CHGNet calculator."""
    from chgnet.model import CHGNetCalculator

    if model:
        return CHGNetCalculator(model_path=model, use_device=device)
    return CHGNetCalculator(use_device=device)
'''

    # Write environment files (use _env suffix to avoid shadowing package names)
    (root / "environments" / "mace_env.py").write_text(mace_env)
    (root / "environments" / "chgnet_env.py").write_text(chgnet_env)

    # Commit volume changes
    rootstock_volume.commit()

    print(f"Initialized rootstock volume at {root}")
    print("\nDirectory structure:")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(root)}")

    return {"status": "initialized", "root": str(root)}


# -----------------------------------------------------------------------------
# v0.3 API Tests
# -----------------------------------------------------------------------------


@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",
    timeout=1800,
)
def test_new_api():
    """Test the v0.3 simplified API with caching."""
    import sys
    import time

    sys.path.insert(0, "/root")

    from ase.build import bulk

    from rootstock import RootstockCalculator

    atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

    # Test 1: cluster + model API (cold cache)
    print("=" * 60)
    print("Test 1: cluster='modal', model='mace-medium' (cold cache)")
    print("=" * 60)

    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="mace-medium",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"Energy: {energy:.6f} eV")
    first_run = time.time() - t0
    print(f"First run time: {first_run:.1f}s")

    # Test 2: Same call again (warm cache)
    print("\n" + "=" * 60)
    print("Test 2: Same call again (warm cache)")
    print("=" * 60)

    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="mace-medium",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"Energy: {energy:.6f} eV")
    second_run = time.time() - t0
    print(f"Second run time: {second_run:.1f}s")

    # Test 3: Different model, same environment
    print("\n" + "=" * 60)
    print("Test 3: model='mace-small' (same env, different model)")
    print("=" * 60)

    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="mace-small",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"Energy: {energy:.6f} eV")
    third_run = time.time() - t0
    print(f"Third run time: {third_run:.1f}s")

    # Test 4: CHGNet
    print("\n" + "=" * 60)
    print("Test 4: model='chgnet'")
    print("=" * 60)

    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="chgnet",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"Energy: {energy:.6f} eV")
    fourth_run = time.time() - t0
    print(f"CHGNet run time: {fourth_run:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"First MACE run (cold):  {first_run:.1f}s")
    print(f"Second MACE run (warm): {second_run:.1f}s")

    speedup = first_run / second_run if second_run > 0 else 0
    print(f"Speedup from caching:   {speedup:.1f}x")

    # Commit any cache changes
    rootstock_volume.commit()

    return {
        "first_run_s": first_run,
        "second_run_s": second_run,
        "cache_speedup": speedup,
    }


@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
)
def inspect_cache():
    """Inspect the cache directory contents."""
    import subprocess
    from pathlib import Path

    root = Path("/vol/rootstock")

    print("=== Directory Structure ===")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            print(f"  {path.relative_to(root)} ({size:,} bytes)")

    print("\n=== Cache Sizes ===")
    hf_cache = root / "cache" / "huggingface"

    if hf_cache.exists():
        result = subprocess.run(
            ["du", "-sh", str(hf_cache)],
            capture_output=True,
            text=True,
        )
        print(f"HuggingFace cache: {result.stdout.strip()}")
    else:
        print("HuggingFace cache: (not created)")

    print("\nNote: uv cache uses local storage (not persisted on volume)")


# -----------------------------------------------------------------------------
# Legacy Environment Tests (backward compatible)
# -----------------------------------------------------------------------------


@app.function(
    image=base_image,
    gpu="A10G",
    timeout=1800,
)
def test_mace_environment(
    model: str = "medium",
    device: str = "cuda",
) -> dict:
    """
    Test the MACE environment file using RootstockCalculator (legacy API).

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
    print("Testing MACE Environment (legacy API)")
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
    image=base_image,
    gpu="A10G",
    timeout=1800,
)
def test_chgnet_environment(
    device: str = "cuda",
) -> dict:
    """Test the CHGNet environment file using RootstockCalculator (legacy API)."""
    import sys
    import time

    sys.path.insert(0, "/root")

    from ase.build import bulk

    from rootstock import RootstockCalculator
    from rootstock.pep723 import validate_environment_file

    env_path = "/root/environments/chgnet_env.py"

    print("=" * 60)
    print("Testing CHGNet Environment (legacy API)")
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
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",
    timeout=3600,
)
def benchmark_v3(
    n_calls: int = 50,
    n_warmup: int = 5,
) -> dict:
    """
    Benchmark RootstockCalculator v0.3 with caching.

    Compares:
    - Direct MACE calculation (baseline)
    - RootstockCalculator with v0.3 API and caching

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
    print("Rootstock v0.3 Benchmark")
    print("=" * 60)
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
            model="medium",
            device="cuda",
        )
        print(direct_result.summary())

        # Rootstock benchmark (v0.3 API)
        print("\n[RootstockCalculator v0.3 - cluster API]")

        times = []
        with RootstockCalculator(
            cluster="modal",
            model="mace-medium",
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

    # Commit cache changes
    rootstock_volume.commit()

    return results


# Legacy benchmark (for backward compatibility)
@app.function(
    image=base_image,
    gpu="A10G",
    timeout=3600,
)
def benchmark_v2(
    n_calls: int = 50,
    n_warmup: int = 5,
    model: str = "medium",
) -> dict:
    """Legacy benchmark using v0.2 API."""
    import sys
    import time

    sys.path.insert(0, "/root")

    import numpy as np

    from benchmarks.direct import benchmark_direct
    from benchmarks.systems import get_benchmark_system
    from rootstock import RootstockCalculator

    print("=" * 60)
    print("Rootstock IPC Benchmark (legacy v0.2 API)")
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

    print("\n" + "=" * 60)
    large_result = results.get("large")
    if large_result:
        if large_result["overhead_pct"] < 5:
            print("SUCCESS: IPC overhead is acceptable for 1000+ atom systems")
        else:
            print("CONCERN: IPC overhead exceeds 5% target for 1000+ atoms")
    print("=" * 60)

    return results


# -----------------------------------------------------------------------------
# Local entry point
# -----------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    """Run the v0.3 test suite."""
    print("Initializing rootstock volume...")
    init_result = init_rootstock_volume.remote()
    print(f"Volume initialized: {init_result}")

    print("\nRunning v0.3 API tests...")
    test_result = test_new_api.remote()

    print("\n\nFinal Results:")
    print("=" * 60)
    print(f"First run (cold cache): {test_result['first_run_s']:.1f}s")
    print(f"Second run (warm cache): {test_result['second_run_s']:.1f}s")
    print(f"Cache speedup: {test_result['cache_speedup']:.1f}x")
