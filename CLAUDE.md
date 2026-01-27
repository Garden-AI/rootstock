# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rootstock is a proof-of-concept for running MLIP (Machine Learning Interatomic Potential) calculators in isolated pre-built Python environments, communicating via the i-PI protocol over Unix sockets.

**Current version: v0.4** - Pre-built environments (no dynamic `uv run`).

## Commands

### Running on Modal
```bash
# Initialize volume and build environments (takes ~10-15 min)
modal run modal_app.py::init_rootstock_volume

# Test pre-built environments
modal run modal_app.py::test_prebuilt

# Show status
modal run modal_app.py::inspect_status

# Run benchmarks
modal run modal_app.py::benchmark_v4
```

### Local Development
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### CLI Commands
```bash
# Build a pre-built environment
rootstock build <env_name> --root <path> [--models m1,m2] [--force]

# Show status
rootstock status --root <path>

# List environments
rootstock list --root <path>
```

### Linting
```bash
ruff check rootstock/
ruff format rootstock/
```

## Architecture

```
Main Process                          Worker Process (subprocess)
┌─────────────────────────┐          ┌─────────────────────────────┐
│ RootstockCalculator     │          │ Pre-built venv Python       │
│ (ASE-compatible)        │          │ (mace_env/bin/python)       │
│                         │          │                             │
│ server.py (i-PI server) │◄────────►│ worker.py (i-PI client)     │
│ - sends positions       │   Unix   │ - receives positions        │
│ - receives forces       │  socket  │ - calculates forces         │
└─────────────────────────┘          └─────────────────────────────┘
```

**Key design**: v0.4 uses pre-built virtual environments instead of dynamic `uv run`. This provides:
- Fast startup (no pip install at runtime)
- Works on any filesystem (no lock files or hardlinks needed)
- Reproducible environments

### Core Files

- `rootstock/cli.py` - CLI commands (`build`, `status`, `list`, `register`)
- `rootstock/calculator.py` - ASE Calculator interface (main entry point)
- `rootstock/server.py` - Spawns worker subprocess, manages socket lifecycle
- `rootstock/worker.py` - i-PI client state machine
- `rootstock/environment.py` - Pre-built environment management, wrapper generation
- `rootstock/clusters.py` - Cluster registry and model string parsing

### Directory Structure

```
{root}/
├── .python/                # uv-managed Python interpreters (portable)
│   └── cpython-3.10.19-linux-x86_64-gnu/
├── environments/           # Environment SOURCE files (*.py with PEP 723)
│   ├── mace_env.py
│   └── chgnet_env.py
├── envs/                   # Pre-built virtual environments
│   ├── mace_env/
│   │   ├── bin/python      # Symlinks to .python/
│   │   ├── lib/python3.11/site-packages/
│   │   └── env_source.py   # Copy of source for imports
│   └── chgnet_env/
└── cache/                  # XDG_CACHE_HOME for model weights
    ├── mace/
    └── huggingface/
```

The entire `{root}/` directory is self-contained and portable. Python interpreters
are stored in `.python/` so venv symlinks resolve correctly on any machine where
the root directory is mounted (Modal Volume, HPC shared filesystem, etc.).

### Known Clusters

| Cluster | Root Path |
|---------|-----------|
| `modal` | `/vol/rootstock` |
| `della` | `/scratch/gpfs/SHARED/rootstock` |

## API

```python
# Standard usage
with RootstockCalculator(
    cluster="modal",      # or root="/path/to/rootstock"
    model="mace-medium",  # -> env_name="mace_env", model_arg="medium"
    device="cuda",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
```

### Model String Format

- `"mace-medium"` → env_name=mace_env, model_arg="medium"
- `"chgnet"` → env_name=chgnet_env, model_arg=""
- `"mace-/path/to/weights.pt"` → env_name=mace_env, model_arg="/path/to/weights.pt"

## Build Process

```bash
# 1. Create environment source file
cat > environments/mace_env.py << 'EOF'
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
EOF

# 2. Build pre-built environment
rootstock build mace_env --root /path/to/rootstock --models small,medium

# 3. Verify
rootstock status --root /path/to/rootstock
```
