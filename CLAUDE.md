# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rootstock is a proof-of-concept for running MLIP (Machine Learning Interatomic Potential) calculators in isolated pre-built Python environments, communicating via the i-PI protocol over Unix sockets.

**Current version: v0.8** - Manifest schema v3; canonical-checkpoint-id API.

## Commands

### Local Development
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### CLI Commands
```bash
# Build a pre-built environment (venv only — no model weights)
rootstock install <env_source.py> [--root <path>] [--force]

# Download + verify a checkpoint by canonical id (idempotent). Use --no-verify on login nodes.
rootstock add <checkpoint-id> [--kwarg key=val ...] [--device cuda] [--no-verify]

# Re-verify all fetched checkpoints (suitable for nightly cron)
rootstock smoke-test [--env ENV] [--checkpoint CKPT] [--device cuda] [--json]

# Show status (per-checkpoint verified/stale grid; --json for machine-readable)
rootstock status [--root <path>] [--json]

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
│ (ASE-compatible)        │          │ (envs/mace/bin/python)      │
│                         │          │                             │
│ server.py (i-PI server) │◄────────►│ worker.py (i-PI client)     │
│ - sends positions       │   Unix   │ - receives positions        │
│ - receives forces       │  socket  │ - calculates forces         │
└─────────────────────────┘          └─────────────────────────────┘
```

**Key design**: pre-built virtual environments instead of dynamic `uv run`. This provides:
- Fast startup (no pip install at runtime)
- Works on any filesystem (no lock files or hardlinks needed)
- Reproducible environments

### Core Files

- `rootstock/cli.py` - CLI commands (`build`, `status`, `list`, `register`)
- `rootstock/calculator.py` - ASE Calculator interface (main entry point)
- `rootstock/server.py` - Spawns worker subprocess, manages socket lifecycle
- `rootstock/worker.py` - i-PI client state machine
- `rootstock/environment.py` - Pre-built environment management, wrapper generation
- `rootstock/clusters.py` - Cluster registry and known environments

### Directory Structure

```
{root}/
├── .python/                # uv-managed Python interpreters (portable)
│   └── cpython-3.10.19-linux-x86_64-gnu/
├── environments/           # Environment SOURCE files (*.py with PEP 723 + CHECKPOINTS)
│   ├── mace.py
│   ├── uma.py
│   └── tensornet.py
├── envs/                   # Pre-built virtual environments
│   ├── mace/
│   │   ├── bin/python      # Symlinks to .python/
│   │   ├── lib/python3.11/site-packages/
│   │   └── env_source.py   # Copy of source for imports
│   └── uma/
└── cache/                  # XDG_CACHE_HOME for model weights
    ├── mace/
    └── huggingface/
```

The install root is self-contained and portable. Python interpreters are stored
in `.python/` so venv symlinks resolve correctly on any machine where the root
is mounted (HPC shared filesystem, NFS, Lustre, etc.).

On clusters where the right filesystem for code is different from the right
filesystem for the model-weight cache (e.g., Perlmutter — code on CFS, cache
on PSCRATCH), the cluster registry encodes both as `root` and `cache_root`.
Most clusters use the same path for both.

### Known Clusters

| Cluster | Install Root | Cache Root (if split) |
|---------|--------------|-----------------------|
| `della` | `/scratch/gpfs/ROSENGROUP/common/rootstock` | (same as install root) |
| `sophia` | `/eagle/Garden-Ai/rootstock` | (same as install root) |
| `polaris` | `/eagle/Garden-Ai/rootstock` (shared with sophia) | (same as install root) |
| `perlmutter` | `/global/cfs/cdirs/m4845/rootstock` | `/pscratch/sd/w/wengler/rootstock-cache` |
| `delta` | `/work/hdd/data/rootstock` | (same as install root) |

## API

```python
# v0.8: single canonical checkpoint id; env is resolved automatically.
with RootstockCalculator(
    cluster="della",
    checkpoint="mace-mp-0-medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()

# Forward extra kwargs to the env's setup() function. Cannot contain
# "checkpoint" or "device" — those are passed at the top level.
with RootstockCalculator(
    cluster="della",
    checkpoint="uma-s-1p1",
    setup_kwargs={"task": "omol"},
) as calc:
    ...
```

### Canonical checkpoint ids (bundled envs)

| Env | Canonical ids |
|---|---|
| `mace` | `mace-mp-0-{small,medium,large}`, `mace-off23-{small,medium,large}` |
| `esen` | `esen-md-direct-all-omol`, `esen-sm-conserving-all-omol`, `esen-sm-direct-all-omol` |
| `orb` | `orb-v2` |
| `tensornet` | `tensornet-matpes-pbe-2025-2` |
| `uma` | `uma-s-1p1` |

## Build Process

```bash
# 1. Create environment source file
cat > environments/mace.py << 'EOF'
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///

CHECKPOINTS = {
    "mace-mp-0-small":  "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large":  "large",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
EOF

# 2. Build pre-built environment (venv only — no model weights)
rootstock install environments/mace.py --root /path/to/rootstock

# 3. Download and verify checkpoints by canonical id (idempotent; use --no-verify on login nodes)
rootstock add mace-mp-0-medium --root /path/to/rootstock
rootstock add mace-mp-0-small --root /path/to/rootstock

# 4. Verify install state
rootstock status --root /path/to/rootstock
```
