# Rootstock v0.3 Design: Environment Caching & Simplified API

## Overview

This document specifies two related improvements for Rootstock v0.3:

1. **Environment dependency caching** — Avoid repeated pip installs by configuring uv's cache directory within the Rootstock root
2. **Simplified user API** — Users specify cluster + model name instead of paths

Target deployments: Modal (testing), Princeton Della (first production user).

---

## 1. Directory Structure

All Rootstock state lives under a single root directory. This enables clean uninstallation (delete one directory) and simplifies HPC administration.

```
{root}/
├── environments/           # Environment files (*.py with PEP 723 metadata)
│   ├── mace.py
│   ├── chgnet.py
│   └── orb.py             # Future
├── cache/
│   ├── uv/                # uv package cache (pip wheels, etc.)
│   └── huggingface/       # HuggingFace model weights
```

### Root Locations by Cluster

Hardcoded in v0.3, extensible later:

| Cluster | Root Path | Notes |
|---------|-----------|-------|
| `modal` | `/vol/rootstock` | Modal persistent volume |
| `della` | `/scratch/gpfs/SHARED/rootstock` | Princeton shared scratch (TBD with admin) |

Users on unlisted clusters pass `root="/path/to/rootstock"` directly.

---

## 2. Environment Caching

### Problem

Currently, `uv run` creates a fresh virtual environment each invocation. On HPC, this means:
- Repeated downloads of torch, mace-torch, etc.
- Slow job startup (minutes instead of seconds)
- Wasted bandwidth and disk I/O

### Solution

Pass `--cache-dir {root}/cache/uv` to `uv run`. This tells uv to reuse cached wheels and built packages across invocations.

### Implementation

Modify `EnvironmentManager.get_spawn_command()`:

```python
def get_spawn_command(self, wrapper_path: Path) -> list[str]:
    """Get the command to spawn a worker via uv run."""
    rootstock_path = get_rootstock_path()
    
    cmd = ["uv", "run"]
    
    # Add cache directory if root is set
    if self.root is not None:
        cache_dir = self.root / "cache" / "uv"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--cache-dir", str(cache_dir)])
    
    cmd.extend(["--with", str(rootstock_path), str(wrapper_path)])
    return cmd
```

### HuggingFace Cache (existing)

Already implemented via `HF_HOME` environment variable in `get_environment_variables()`. Ensure it points to `{root}/cache/huggingface`.

### Cache Warming (optional, future)

For first-time setup on a new cluster, an admin could run:

```bash
rootstock warm --root /scratch/gpfs/SHARED/rootstock
```

This would pre-download all registered environments' dependencies. Not required for v0.3.

---

## 3. Simplified API

### Current API (v0.2)

```python
with RootstockCalculator(
    environment="/path/to/environments/mace.py",
    model="medium",
    device="cuda",
    root="/path/to/rootstock",
) as calc:
    ...
```

Problems:
- User must know paths
- Easy to mismatch environment file and root directory
- Verbose

### New API (v0.3)

```python
# Preferred: cluster + model
with RootstockCalculator(
    cluster="della",
    model="mace-medium",
    device="cuda",
) as calc:
    ...

# Alternative: explicit root + model
with RootstockCalculator(
    root="/scratch/gpfs/SHARED/rootstock",
    model="mace-medium", 
    device="cuda",
) as calc:
    ...

# Power user: full manual control (backwards compatible)
with RootstockCalculator(
    environment="/custom/path/mace.py",
    model="medium",
    device="cuda",
) as calc:
    ...
```

### Model String Format

The `model` parameter encodes both the environment and the model-specific argument:

```
model = "{environment_name}" | "{environment_name}-{model_arg}"
```

Examples:

| `model=` | Environment File | `setup()` model arg |
|----------|------------------|---------------------|
| `"mace-small"` | `mace.py` | `"small"` |
| `"mace-medium"` | `mace.py` | `"medium"` |
| `"mace-large"` | `mace.py` | `"large"` |
| `"mace-/path/to/custom.pt"` | `mace.py` | `"/path/to/custom.pt"` |
| `"chgnet"` | `chgnet.py` | `""` (empty, use default) |
| `"chgnet-0.3.0"` | `chgnet.py` | `"0.3.0"` (if CHGNet supports version selection) |

### Parsing Logic

```python
def parse_model_string(model: str) -> tuple[str, str]:
    """
    Parse model string into (environment_name, model_arg).
    
    Returns:
        (environment_name, model_arg) tuple
    """
    # Check for known environment prefixes
    known_envs = ["mace", "chgnet", "orb", "alignn"]  # Extend as needed
    
    for env in known_envs:
        if model == env:
            return (env, "")
        if model.startswith(f"{env}-"):
            model_arg = model[len(env) + 1:]  # Everything after "env-"
            return (env, model_arg)
    
    # Unknown format - treat entire string as environment name
    return (model, "")
```

### Cluster Registry

```python
# rootstock/clusters.py

CLUSTER_REGISTRY: dict[str, str] = {
    "modal": "/vol/rootstock",
    "della": "/scratch/gpfs/SHARED/rootstock",
}

def get_root_for_cluster(cluster: str) -> Path:
    """Get the rootstock root directory for a known cluster."""
    if cluster not in CLUSTER_REGISTRY:
        available = ", ".join(CLUSTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown cluster '{cluster}'. Known clusters: {available}. "
            f"Use root='/path/to/rootstock' for custom locations."
        )
    return Path(CLUSTER_REGISTRY[cluster])
```

### Updated RootstockCalculator.__init__

```python
def __init__(
    self,
    # New v0.3 parameters
    model: str = "mace-medium",
    cluster: Optional[str] = None,
    # Existing parameters
    root: Optional[str | Path] = None,
    device: str = "cuda",
    # Legacy/power-user parameters
    environment: Optional[str] = None,
    **kwargs,
):
    """
    Initialize the Rootstock calculator.
    
    Args:
        model: Model identifier, e.g. "mace-medium", "chgnet", "mace-/path/to/weights.pt"
        cluster: Known cluster name ("modal", "della"). Mutually exclusive with root.
        root: Path to rootstock directory. Mutually exclusive with cluster.
        device: PyTorch device ("cuda", "cuda:0", "cpu")
        environment: Direct path to environment file (power user, overrides model parsing)
    """
    super().__init__(**kwargs)
    
    # Resolve root directory
    if cluster is not None and root is not None:
        raise ValueError("Cannot specify both 'cluster' and 'root'")
    
    if cluster is not None:
        self.root = get_root_for_cluster(cluster)
    elif root is not None:
        self.root = Path(root)
    else:
        raise ValueError("Must specify either 'cluster' or 'root'")
    
    # Resolve environment and model arg
    if environment is not None:
        # Power user mode: direct environment path
        self.environment_path = Path(environment)
        self.model_arg = model  # Pass model string directly to setup()
    else:
        # Standard mode: parse model string
        env_name, model_arg = parse_model_string(model)
        self.environment_path = self.root / "environments" / f"{env_name}.py"
        self.model_arg = model_arg
        
        if not self.environment_path.exists():
            available = list((self.root / "environments").glob("*.py"))
            raise FileNotFoundError(
                f"Environment '{env_name}' not found at {self.environment_path}. "
                f"Available: {[p.stem for p in available]}"
            )
    
    self.device = device
    # ... rest of init
```

---

## 4. Modal Volume Testing

### Setup

Create a persistent Modal volume for testing:

```python
# modal_app.py

import modal

app = modal.App("rootstock-v03-test")

# Persistent volume for rootstock root directory
rootstock_volume = modal.Volume.from_name("rootstock-data", create_if_missing=True)

# Base image with uv
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .pip_install("torch>=2.0", "numpy>=1.24", "ase>=3.22", "tomli>=2.0")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir("rootstock", "/root/rootstock")
    .add_local_file("pyproject.toml", "/root/pyproject.toml")
)
```

### Initialize Volume Structure

```python
@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    timeout=600,
)
def init_rootstock_volume():
    """One-time setup: create directory structure and copy environment files."""
    from pathlib import Path
    import shutil
    
    root = Path("/vol/rootstock")
    
    # Create directory structure
    (root / "environments").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "uv").mkdir(parents=True, exist_ok=True)
    (root / "cache" / "huggingface").mkdir(parents=True, exist_ok=True)
    
    # Copy environment files (these would be bundled with the image or fetched)
    # For testing, create them inline:
    
    mace_env = '''# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///
"""MACE environment for Rootstock."""

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
'''
    
    chgnet_env = '''# /// script
# requires-python = ">=3.10"  
# dependencies = ["chgnet>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///
"""CHGNet environment for Rootstock."""

def setup(model: str | None = None, device: str = "cuda"):
    from chgnet.model import CHGNetCalculator
    if model:
        return CHGNetCalculator(model_path=model, use_device=device)
    return CHGNetCalculator(use_device=device)
'''
    
    (root / "environments" / "mace.py").write_text(mace_env)
    (root / "environments" / "chgnet.py").write_text(chgnet_env)
    
    # Commit volume changes
    rootstock_volume.commit()
    
    print(f"Initialized rootstock volume at {root}")
    print(f"Contents: {list(root.rglob('*'))}")
```

### Test New API

```python
@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",
    timeout=1800,
)
def test_new_api():
    """Test the v0.3 simplified API."""
    import sys
    import time
    sys.path.insert(0, "/root")
    
    from ase.build import bulk
    from rootstock import RootstockCalculator
    
    atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)
    
    # Test 1: cluster + model API
    print("=" * 60)
    print("Test 1: cluster='modal', model='mace-medium'")
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
    print(f"First run (cold cache): {first_run:.1f}s")
    
    # Test 2: Same thing again - should be faster (warm cache)
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
    print(f"Second run (warm cache): {second_run:.1f}s")
    
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
    print(f"Third run (env cached, new model): {third_run:.1f}s")
    
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
    print(f"CHGNet run: {fourth_run:.1f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"First MACE run (cold):  {first_run:.1f}s")
    print(f"Second MACE run (warm): {second_run:.1f}s")
    print(f"Speedup from caching:   {first_run/second_run:.1f}x")
    
    # Commit any cache changes
    rootstock_volume.commit()
    
    return {
        "first_run_s": first_run,
        "second_run_s": second_run,
        "cache_speedup": first_run / second_run,
    }
```

### Verify Cache Contents

```python
@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
)
def inspect_cache():
    """Inspect the cache directory contents."""
    from pathlib import Path
    import subprocess
    
    root = Path("/vol/rootstock")
    
    print("=== Directory Structure ===")
    subprocess.run(["find", str(root), "-type", "f"], check=True)
    
    print("\n=== Cache Sizes ===")
    subprocess.run(["du", "-sh", str(root / "cache" / "uv")], check=True)
    subprocess.run(["du", "-sh", str(root / "cache" / "huggingface")], check=True)
```

---

## 5. Migration Path

### For Users

v0.2 code continues to work unchanged. The `environment=` parameter is still supported.

Migration is optional but recommended:

```python
# v0.2 (still works)
with RootstockCalculator(
    environment="/scratch/gpfs/SHARED/rootstock/environments/mace.py",
    model="medium",
    root="/scratch/gpfs/SHARED/rootstock",
    device="cuda",
) as calc:
    ...

# v0.3 (recommended)
with RootstockCalculator(
    cluster="della",
    model="mace-medium",
    device="cuda",
) as calc:
    ...
```

### For Administrators

1. Create the rootstock directory structure on the cluster
2. Copy/register environment files
3. (Optional) Warm the cache by running each environment once

---

## 6. Implementation Checklist

- [ ] Add `rootstock/clusters.py` with `CLUSTER_REGISTRY` and `get_root_for_cluster()`
- [ ] Add `parse_model_string()` function  
- [ ] Update `RootstockCalculator.__init__` with new parameter handling
- [ ] Update `EnvironmentManager.get_spawn_command()` to pass `--cache-dir`
- [ ] Update `EnvironmentManager.get_environment_variables()` to ensure `UV_CACHE_DIR` is set (belt and suspenders)
- [ ] Add Modal volume setup in `modal_app.py`
- [ ] Add tests for new API
- [ ] Update README with new usage examples
- [ ] Update CONTEXT.md with current state

---

## 7. Future Considerations (Not v0.3)

- **User-defined clusters**: `~/.config/rootstock/clusters.toml`
- **Environment versioning**: `mace@2024.01.15.py`
- **Cache warming command**: `rootstock warm --root /path`
- **Environment discovery**: `rootstock list --cluster della`
- **Health checks**: `rootstock doctor --cluster della`