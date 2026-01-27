"""
Modal app for running Rootstock v0.4 tests and benchmarks.

Usage:
    # Initialize the volume (build environments - takes ~10-15 min)
    modal run modal_app.py::init_rootstock_volume

    # Test pre-built environments
    modal run modal_app.py::test_prebuilt

    # Inspect status
    modal run modal_app.py::inspect_status

    # Run benchmarks
    modal run modal_app.py::benchmark_v4
"""

import modal

# -----------------------------------------------------------------------------
# Modal configuration
# -----------------------------------------------------------------------------

app = modal.App("rootstock-v04")

# Persistent volume for rootstock root directory
rootstock_volume = modal.Volume.from_name("rootstock-v04-data", create_if_missing=True)

# Base image with uv and Python
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

# Environment source files
MACE_ENV_SOURCE = '''# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///
"""MACE environment for Rootstock."""


def setup(model: str, device: str = "cuda"):
    """Load a MACE calculator."""
    from mace.calculators import mace_mp

    return mace_mp(model=model, device=device, default_dtype="float32")
'''

CHGNET_ENV_SOURCE = '''# /// script
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


# -----------------------------------------------------------------------------
# Volume Initialization
# -----------------------------------------------------------------------------


@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",  # Need GPU for model downloads that require CUDA
    timeout=3600,  # Building envs takes time
)
def init_rootstock_volume():
    """One-time setup: create structure, write sources, build environments."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")

    root = Path("/vol/rootstock")

    # Clean up old files
    for subdir in ["environments", "envs", "cache", ".python"]:
        path = root / subdir
        if path.exists():
            shutil.rmtree(path)

    # Create directory structure
    (root / "environments").mkdir(parents=True, exist_ok=True)
    (root / "envs").mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)

    # Write environment source files
    (root / "environments" / "mace_env.py").write_text(MACE_ENV_SOURCE)
    (root / "environments" / "chgnet_env.py").write_text(CHGNET_ENV_SOURCE)

    print("Environment sources written")

    # Build environments using rootstock CLI
    for env_name, models in [("mace_env", "small,medium"), ("chgnet_env", None)]:
        print(f"\n{'='*60}")
        print(f"Building {env_name}...")
        print("=" * 60)

        cmd = [
            sys.executable, "-m", "rootstock.cli",
            "build", env_name,
            "--root", str(root),
            "--verbose",
        ]
        if models:
            cmd.extend(["--models", models])

        result = subprocess.run(cmd, cwd="/root")
        if result.returncode != 0:
            print(f"Warning: Failed to build {env_name}")

        # Commit volume after each environment build
        print(f"\nCommitting volume after {env_name}...")
        rootstock_volume.commit()

    # Show status
    print("\n" + "=" * 60)
    print("Final status:")
    print("=" * 60)
    subprocess.run(
        [sys.executable, "-m", "rootstock.cli", "status", "--root", str(root)],
        cwd="/root",
    )

    # Final commit
    print("\nFinal volume commit...")
    rootstock_volume.commit()

    return {"status": "initialized", "root": str(root)}


# -----------------------------------------------------------------------------
# v0.4 Tests
# -----------------------------------------------------------------------------


@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",
    timeout=600,
)
def test_prebuilt():
    """Test that pre-built environments work and are fast."""
    import sys
    import time

    sys.path.insert(0, "/root")

    from ase.build import bulk

    from rootstock import RootstockCalculator

    atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

    print("=" * 60)
    print("Testing Pre-built Environments (v0.4)")
    print("=" * 60)

    # Test 1: First run - should be fast since env is pre-built
    print("\nTest 1: First run with mace-medium")
    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="mace-medium",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"  Energy: {energy:.6f} eV")
    first_run = time.time() - t0
    print(f"  Time: {first_run:.1f}s")

    # Test 2: Second run - should be similarly fast
    print("\nTest 2: Second run (same model)")
    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="mace-medium",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"  Energy: {energy:.6f} eV")
    second_run = time.time() - t0
    print(f"  Time: {second_run:.1f}s")

    # Test 3: Different model
    print("\nTest 3: Different model (mace-small)")
    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="mace-small",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"  Energy: {energy:.6f} eV")
    third_run = time.time() - t0
    print(f"  Time: {third_run:.1f}s")

    # Test 4: CHGNet
    print("\nTest 4: CHGNet")
    t0 = time.time()
    with RootstockCalculator(
        cluster="modal",
        model="chgnet",
        device="cuda",
    ) as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"  Energy: {energy:.6f} eV")
    chgnet_run = time.time() - t0
    print(f"  Time: {chgnet_run:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"First run:   {first_run:.1f}s")
    print(f"Second run:  {second_run:.1f}s")
    print(f"mace-small:  {third_run:.1f}s")
    print(f"CHGNet:      {chgnet_run:.1f}s")

    # Both should be fast (< 30s) since no env creation needed
    if first_run < 60 and second_run < 60:
        print("\nSUCCESS: Pre-built environments are fast!")
    else:
        print("\nWARNING: Runs are slow, something may be wrong")

    return {
        "first_run_s": first_run,
        "second_run_s": second_run,
        "mace_small_s": third_run,
        "chgnet_s": chgnet_run,
    }


@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
)
def inspect_status():
    """Show rootstock status and check Python symlinks."""
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")

    subprocess.run(
        [sys.executable, "-m", "rootstock.cli", "status", "--root", "/vol/rootstock"],
        cwd="/root",
    )

    # Check Python symlinks and test execution
    root = Path("/vol/rootstock")
    print("\nPython symlink check:")
    for env_name in ["mace_env", "chgnet_env"]:
        python_bin = root / "envs" / env_name / "bin" / "python"
        if python_bin.exists():
            if python_bin.is_symlink():
                target = python_bin.readlink()
                resolved = python_bin.resolve()
                exists = resolved.exists()
                print(f"  {env_name}: {target}")
                print(f"    resolved: {resolved} (exists={exists})")

                # Test execution
                import time
                t0 = time.time()
                result = subprocess.run(
                    [str(python_bin), "-c", "print('hello')"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                print(f"    exec test: {result.stdout.strip()} ({time.time()-t0:.1f}s)")
            else:
                print(f"  {env_name}: not a symlink")
        else:
            print(f"  {env_name}: python not found")


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------


@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",
    timeout=3600,
)
def benchmark_v4(
    n_calls: int = 50,
    n_warmup: int = 5,
) -> dict:
    """
    Benchmark RootstockCalculator v0.4 with pre-built environments.

    Compares:
    - Direct MACE calculation (baseline)
    - RootstockCalculator with pre-built environment
    """
    import sys
    import time

    sys.path.insert(0, "/root")

    import numpy as np

    from benchmarks.direct import benchmark_direct
    from benchmarks.systems import get_benchmark_system
    from rootstock import RootstockCalculator

    print("=" * 60)
    print("Rootstock v0.4 Benchmark (Pre-built Environments)")
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

        # Rootstock benchmark (v0.4 API with pre-built env)
        print("\n[RootstockCalculator v0.4 - pre-built env]")

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
    """Run the v0.4 test suite."""
    print("Initializing rootstock volume (this may take 10-15 minutes)...")
    init_result = init_rootstock_volume.remote()
    print(f"Volume initialized: {init_result}")

    print("\nRunning pre-built environment tests...")
    test_result = test_prebuilt.remote()

    print("\n\nFinal Results:")
    print("=" * 60)
    print(f"First run:  {test_result['first_run_s']:.1f}s")
    print(f"Second run: {test_result['second_run_s']:.1f}s")
