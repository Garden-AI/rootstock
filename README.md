# Rootstock

Run MLIP (Machine Learning Interatomic Potential) calculators in isolated pre-built Python environments, communicating via the i-PI protocol over Unix sockets.

## Quick Start

```python
from ase.build import bulk
from rootstock import RootstockCalculator

atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

# Using a known cluster
with RootstockCalculator(
    cluster="modal",       # or "della"
    model="mace-medium",   # or "chgnet", "mace-small", etc.
    device="cuda",
) as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())

# Or with an explicit root path
with RootstockCalculator(
    root="/scratch/gpfs/SHARED/rootstock",
    model="mace-medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy())
```

**Note:** Environments must be pre-built before use. See [Administrator Setup](#administrator-setup).

## Installation

```bash
pip install rootstock
# or
uv pip install rootstock
```

## Model String Format

The `model` parameter encodes both the environment and model-specific argument:

| `model=`            | Environment    | Model Arg           |
|---------------------|----------------|---------------------|
| `"mace-medium"`     | mace_env       | `"medium"`          |
| `"mace-small"`      | mace_env       | `"small"`           |
| `"mace-large"`      | mace_env       | `"large"`           |
| `"chgnet"`          | chgnet_env     | `""` (default)      |
| `"mace-/path/to/weights.pt"` | mace_env | `"/path/to/weights.pt"` |

## Known Clusters

| Cluster | Root Path |
|---------|-----------|
| `modal` | `/vol/rootstock` |
| `della` | `/scratch/gpfs/SHARED/rootstock` |

For other clusters, use `root="/path/to/rootstock"` directly.

## Administrator Setup

Environments must be pre-built before users can run calculations.

### 1. Create Directory Structure

```bash
mkdir -p /scratch/gpfs/SHARED/rootstock/{environments,envs,cache}
```

### 2. Create Environment Source Files

```bash
# mace_env.py
cat > /scratch/gpfs/SHARED/rootstock/environments/mace_env.py << 'EOF'
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///
"""MACE environment for Rootstock."""

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
EOF
```

### 3. Build Environments

```bash
# Build MACE environment with model pre-download
rootstock build mace_env --root /scratch/gpfs/SHARED/rootstock --models small,medium,large

# Build CHGNet environment
rootstock build chgnet_env --root /scratch/gpfs/SHARED/rootstock

# Verify
rootstock status --root /scratch/gpfs/SHARED/rootstock
```

## Architecture

```
Main Process                          Worker Process (subprocess)
+-------------------------+          +-----------------------------+
| RootstockCalculator     |          | Pre-built venv Python       |
| (ASE-compatible)        |          | (mace_env/bin/python)       |
|                         |          |                             |
| server.py (i-PI server) |<-------->| worker.py (i-PI client)     |
| - sends positions       |   Unix   | - receives positions        |
| - receives forces       |  socket  | - calculates forces         |
+-------------------------+          +-----------------------------+
```

The worker process uses a pre-built virtual environment, providing:
- **Fast startup**: No dependency installation at runtime
- **Filesystem compatibility**: Works on NFS, Lustre, GPFS, Modal volumes
- **Reproducibility**: Same environment every time

## Directory Structure

```
{root}/
├── environments/           # Environment SOURCE files (*.py with PEP 723)
│   ├── mace_env.py
│   └── chgnet_env.py
├── envs/                   # Pre-built virtual environments
│   ├── mace_env/
│   │   ├── bin/python
│   │   ├── lib/python3.11/site-packages/
│   │   └── env_source.py   # Copy of environment source
│   └── chgnet_env/
└── cache/                  # XDG_CACHE_HOME for model weights
    ├── mace/               # MACE models
    └── huggingface/        # HuggingFace models
```

## CLI Commands

```bash
# Build a pre-built environment
rootstock build <env_name> --root <path> [--models m1,m2] [--force]

# Show status
rootstock status --root <path>

# Register an environment source file
rootstock register <env_file> --root <path>

# List environments
rootstock list --root <path>
```

## Running on Modal

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

## Performance

IPC overhead is <5% for systems with 1000+ atoms compared to direct in-process execution.

| System Size | Atoms | Typical Overhead |
|-------------|-------|------------------|
| Small       | 64    | ~10-15%          |
| Medium      | 256   | ~5-8%            |
| Large       | 1000  | <5%              |

## Local Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check rootstock/
ruff format rootstock/
```
