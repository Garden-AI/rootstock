# Rootstock v0.4 Design: Pre-built Environments

## Overview

Replace dynamic `uv run` environment creation with pre-built virtual environments. This solves filesystem compatibility issues (locking, hardlinks) and provides predictable, fast worker startup.

**Key insight**: The v0.3 `uv run` approach fights the filesystem at runtime. Pre-built environments move all that complexity to a one-time admin build step, leaving runtime as pure read-only access.

---

## 1. Problem Statement

### Why v0.3 Dynamic Environments Don't Work

When a worker runs `uv run --with rootstock wrapper.py`:

1. **Lock files** — uv uses file locks to prevent cache corruption. Many network filesystems (NFS, Lustre, GPFS) and Modal volumes don't support POSIX locking reliably.

2. **Hardlinks** — uv hardlinks cached wheels into environments. Network filesystems often don't support cross-directory hardlinks or have restrictions.

3. **Concurrent access** — Multiple workers spawning simultaneously may corrupt shared cache state.

4. **Unpredictable timing** — First worker pays full install cost; subsequent workers may or may not benefit from caching depending on filesystem behavior.

### Why Pre-built Environments Work

A pre-built environment is a directory tree that already exists. At runtime:

- Worker runs `/path/to/env/bin/python wrapper.py`
- Python reads `.py` and `.so` files (all reads, no writes)
- No locking, no coordination, no cache management

Network filesystems handle concurrent reads perfectly—that's their primary use case.

---

## 2. Directory Structure

```
{root}/
├── environments/           # Environment SOURCE files (*.py with PEP 723)
│   ├── mace_env.py
│   └── chgnet_env.py
├── envs/                   # Pre-built virtual environments
│   ├── mace_env/
│   │   ├── bin/python
│   │   ├── lib/python3.11/site-packages/
│   │   └── pyvenv.cfg
│   └── chgnet_env/
│       └── ...
└── cache/                  # XDG_CACHE_HOME points here
    ├── mace/               # MACE models (auto-created by mace_mp())
    └── huggingface/        # HF Hub models
        └── hub/
```

### Key Points

- **`environments/`** contains the source files (what we have today)
- **`envs/`** contains fully-built virtual environments (new)
- **`cache/`** is set as `XDG_CACHE_HOME`, so libraries create their standard subdirectories (e.g., `mace/`, `huggingface/`)

---

## 3. Build Process

### New CLI Command

```bash
# Build a single environment
rootstock build mace_env --root /path/to/rootstock

# Build with specific models pre-downloaded
rootstock build mace_env --root /path/to/rootstock --models small,medium,large

# Build all registered environments
rootstock build --all --root /path/to/rootstock

# Rebuild (delete and recreate)
rootstock build mace_env --root /path/to/rootstock --force
```

### What `rootstock build` Does

```python
def cmd_build(env_name: str, root: Path, models: list[str] | None = None):
    """Build a pre-built environment."""
    
    env_source = root / "environments" / f"{env_name}.py"
    env_target = root / "envs" / env_name
    
    # 1. Parse PEP 723 metadata from source file
    metadata = parse_pep723_metadata(env_source.read_text())
    dependencies = metadata["dependencies"]
    python_version = metadata.get("requires-python", ">=3.10")
    
    # 2. Create virtual environment
    if env_target.exists():
        raise RuntimeError(f"Environment exists: {env_target}. Use --force to rebuild.")
    
    subprocess.run(["uv", "venv", str(env_target), "--python", python_version])
    
    # 3. Install dependencies + rootstock
    pip = env_target / "bin" / "pip"
    subprocess.run([str(pip), "install", *dependencies])
    subprocess.run([str(pip), "install", "rootstock"])  # Install ourselves
    
    # 4. Copy environment source file into the env for imports
    shutil.copy(env_source, env_target / "env_source.py")
    
    # 5. Pre-download model weights (optional)
    if models:
        download_models(env_target, env_source, models, root / "cache" / "models")
    
    print(f"Built environment: {env_target}")
```

### Model Weight Pre-download

Each MLIP has its own cache mechanism. What we know from actual library documentation:

| Library | Cache Location | Controllable Via |
|---------|---------------|------------------|
| MACE | `~/.cache/mace/` | `XDG_CACHE_HOME` (standard) |
| HuggingFace Hub | `~/.cache/huggingface/` | `HF_HOME`, `HF_HUB_CACHE` |

**Strategy**: Set `XDG_CACHE_HOME` to redirect the standard `~/.cache/` base directory. This catches MACE and any other libraries following the XDG Base Directory spec. Additionally set `HF_HOME` for HuggingFace. We'll document additional env vars per-MLIP as we discover them.

```python
def get_model_cache_env(root: Path) -> dict[str, str]:
    """Environment variables to redirect model downloads to shared cache.
    
    Note: We set XDG_CACHE_HOME as a catch-all for libraries following the
    XDG spec (MACE uses ~/.cache/mace/), plus explicit HF vars for HuggingFace.
    Additional MLIPs may need their own env vars - document as discovered.
    """
    cache_dir = root / "cache"
    return {
        # XDG base directory - catches MACE and other well-behaved libraries
        "XDG_CACHE_HOME": str(cache_dir),
        
        # HuggingFace explicit (some tools check these before XDG)
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
    }
```

To pre-download, we actually invoke `setup()` with `device="cpu"`:

```python
def download_models(env_target: Path, env_source: Path, models: list[str], root: Path):
    """Pre-download model weights by invoking setup()."""
    
    python = env_target / "bin" / "python"
    env_vars = get_model_cache_env(root)
    
    for model in models:
        # Run setup(model, "cpu") to trigger download without needing GPU
        script = f'''
import sys
sys.path.insert(0, "{env_target}")
from env_source import setup
calc = setup("{model}", "cpu")
print(f"Downloaded model: {model}")
'''
        subprocess.run(
            [str(python), "-c", script],
            env={**os.environ, **env_vars},
            check=True,
        )
```

---

## 4. Runtime Changes

### Worker Spawning (Updated)

Instead of `uv run`, spawn directly with the pre-built environment's Python:

```python
# v0.3 (dynamic)
cmd = ["uv", "run", "--with", rootstock_path, wrapper_path]

# v0.4 (pre-built)
env_python = self.root / "envs" / env_name / "bin" / "python"
cmd = [str(env_python), str(wrapper_path)]
```

### Updated `EnvironmentManager`

```python
class EnvironmentManager:
    def __init__(self, root: Path):
        self.root = root
    
    def get_env_python(self, env_name: str) -> Path:
        """Get path to Python executable for a pre-built environment."""
        env_python = self.root / "envs" / env_name / "bin" / "python"
        
        if not env_python.exists():
            available = [p.name for p in (self.root / "envs").iterdir() if p.is_dir()]
            raise RuntimeError(
                f"Environment '{env_name}' not built. "
                f"Run: rootstock build {env_name} --root {self.root}\n"
                f"Available environments: {available}"
            )
        
        return env_python
    
    def get_spawn_command(self, env_name: str, wrapper_path: Path) -> list[str]:
        """Get command to spawn a worker."""
        env_python = self.get_env_python(env_name)
        return [str(env_python), str(wrapper_path)]
    
    def get_environment_variables(self) -> dict[str, str]:
        """Environment variables for worker process."""
        env = os.environ.copy()
        env.update(get_model_cache_env(self.root))
        return env
```

### Wrapper Script (Simplified)

Since rootstock is now installed in the environment, the wrapper is simpler:

```python
WRAPPER_TEMPLATE = '''
import sys
sys.path.insert(0, "{env_dir}")
from env_source import setup
from rootstock.worker import run_worker

run_worker(
    setup_fn=setup,
    model="{model}",
    device="{device}",
    socket_path="{socket_path}",
)
'''
```

Note: No PEP 723 metadata needed—dependencies are already installed.

---

## 5. Updated Calculator Flow

```python
class RootstockCalculator(Calculator):
    def __init__(
        self,
        model: str = "mace-medium",
        cluster: str | None = None,
        root: Path | None = None,
        device: str = "cuda",
        **kwargs,
    ):
        # Resolve root from cluster or explicit path
        if cluster is not None:
            self.root = get_root_for_cluster(cluster)
        elif root is not None:
            self.root = Path(root)
        else:
            raise ValueError("Must specify 'cluster' or 'root'")
        
        # Parse model string: "mace-medium" -> ("mace_env", "medium")
        env_name, model_arg = parse_model_string(model)
        
        # Verify environment is built
        env_python = self.root / "envs" / env_name / "bin" / "python"
        if not env_python.exists():
            raise RuntimeError(
                f"Environment '{env_name}' not built at {self.root}/envs/{env_name}/\n"
                f"Run: rootstock build {env_name} --root {self.root}"
            )
        
        self.env_name = env_name
        self.model_arg = model_arg
        self.device = device
```

---

## 6. Modal Testing Strategy

### Volume Initialization

```python
@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    timeout=1800,  # Building envs takes time
)
def init_rootstock_volume():
    """One-time setup: create structure, copy sources, build environments."""
    from pathlib import Path
    
    root = Path("/vol/rootstock")
    
    # Create directory structure
    (root / "environments").mkdir(parents=True, exist_ok=True)
    (root / "envs").mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)  # XDG_CACHE_HOME
    
    # Write environment source files
    (root / "environments" / "mace_env.py").write_text(MACE_ENV_SOURCE)
    (root / "environments" / "chgnet_env.py").write_text(CHGNET_ENV_SOURCE)
    
    # Build environments (this is the slow part - ~10-15 min each)
    import subprocess
    
    for env_name, models in [("mace_env", ["small", "medium"]), ("chgnet_env", [])]:
        print(f"Building {env_name}...")
        cmd = ["rootstock", "build", env_name, "--root", str(root)]
        if models:
            cmd.extend(["--models", ",".join(models)])
        subprocess.run(cmd, check=True)
    
    rootstock_volume.commit()
    print("Volume initialized with pre-built environments")
```

### Testing Pre-built Environments

```python
@app.function(
    image=base_image,
    volumes={"/vol/rootstock": rootstock_volume},
    gpu="A10G",
    timeout=600,
)
def test_prebuilt():
    """Test that pre-built environments work and are fast."""
    import time
    from rootstock import RootstockCalculator
    from ase.build import bulk
    
    atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)
    
    # First run - should be fast since env is pre-built
    t0 = time.time()
    with RootstockCalculator(cluster="modal", model="mace-medium") as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
    first_run = time.time() - t0
    
    # Second run - should be similarly fast
    t0 = time.time()
    with RootstockCalculator(cluster="modal", model="mace-medium") as calc:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
    second_run = time.time() - t0
    
    print(f"First run:  {first_run:.1f}s")
    print(f"Second run: {second_run:.1f}s")
    
    # Both should be fast (< 30s) since no env creation needed
    # The time is just model loading + calculation
    assert first_run < 60, f"First run too slow: {first_run}s"
    assert abs(first_run - second_run) < 10, "Runs should be similar speed"
```

---

## 7. CLI Summary

```bash
# Build commands (admin)
rootstock build <env_name> --root <path> [--models m1,m2] [--force]
rootstock build --all --root <path>

# Existing commands (unchanged)
rootstock test <env_file> --model <model> --root <path>
rootstock register <env_file> --root <path>
rootstock list --root <path>

# New inspection commands
rootstock status --root <path>              # Show what's built, cache sizes
rootstock doctor --root <path>              # Verify everything is healthy
```

### Example `rootstock status` Output

```
Rootstock root: /scratch/gpfs/SHARED/rootstock

Environments:
  mace_env     source: ✓  built: ✓  models: small, medium, large
  chgnet_env   source: ✓  built: ✓  models: (default)
  orb_env      source: ✓  built: ✗  

Cache (XDG_CACHE_HOME):
  mace/:        1.2 GB
  huggingface/: 890 MB
  Total:        2.1 GB
```

---

## 8. Migration from v0.3

### For Users

No API changes. `RootstockCalculator(cluster="della", model="mace-medium")` works identically.

The only difference: if the environment isn't pre-built, v0.3 would try (and likely fail) with `uv run`. v0.4 fails fast with a clear error message.

### For Administrators

One-time setup:

```bash
# Create root directory
mkdir -p /scratch/gpfs/SHARED/rootstock/{environments,envs,cache}

# Copy/create environment source files
cp mace_env.py chgnet_env.py /scratch/gpfs/SHARED/rootstock/environments/

# Build environments (do this on a build node, not login node)
rootstock build mace_env --root /scratch/gpfs/SHARED/rootstock --models small,medium,large
rootstock build chgnet_env --root /scratch/gpfs/SHARED/rootstock

# Verify
rootstock status --root /scratch/gpfs/SHARED/rootstock
```

---

## 9. Implementation Checklist

- [ ] Add `rootstock build` CLI command
- [ ] Add `get_model_cache_env()` with `XDG_CACHE_HOME` and `HF_HOME`/`HF_HUB_CACHE`
- [ ] Update `EnvironmentManager` to use pre-built envs
- [ ] Update `RootstockServer._start_worker()` to use env Python directly
- [ ] Simplify wrapper template (no PEP 723 needed)
- [ ] Add `rootstock status` command
- [ ] Update Modal volume init to build environments
- [ ] Add tests for pre-built environment flow
- [ ] Update README with admin setup instructions
- [ ] Update CLAUDE.md with new architecture

---

## 10. Open Questions

1. **Should rootstock be pinned in built envs?** If a user has rootstock 0.4.1 but the env was built with 0.4.0, will there be issues? Probably not for the worker protocol, but worth considering.

2. **How to handle Python version mismatches?** If the environment source says `requires-python = ">=3.10"` but the cluster only has 3.9, the build will fail. This is probably the right behavior (fail early), but error messages should be clear.