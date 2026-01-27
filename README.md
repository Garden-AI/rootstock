# Rootstock

Run MLIP (Machine Learning Interatomic Potential) calculators in isolated Python environments, communicating via the i-PI protocol over Unix sockets.

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
| `"mace-medium"`     | mace_env.py    | `"medium"`          |
| `"mace-small"`      | mace_env.py    | `"small"`           |
| `"mace-large"`      | mace_env.py    | `"large"`           |
| `"chgnet"`          | chgnet_env.py  | `""` (default)      |
| `"mace-/path/to/weights.pt"` | mace_env.py | `"/path/to/weights.pt"` |

## Known Clusters

| Cluster | Root Path |
|---------|-----------|
| `modal` | `/vol/rootstock` |
| `della` | `/scratch/gpfs/SHARED/rootstock` |

For other clusters, use `root="/path/to/rootstock"` directly.

## Architecture

```
Main Process                          Worker Process (subprocess)
+-------------------------+          +-----------------------------+
| RootstockCalculator     |          | MLIPWorker                  |
| (ASE-compatible)        |          | (loads MACE/CHGNet once)    |
|                         |          |                             |
| server.py (i-PI server) |<-------->| worker.py (i-PI client)     |
| - sends positions       |   Unix   | - receives positions        |
| - receives forces       |  socket  | - calculates forces         |
+-------------------------+          +-----------------------------+
```

The worker process is spawned via `uv run` with the appropriate environment file, which specifies dependencies via PEP 723 metadata. This enables true Python environment isolation while maintaining a persistent worker across calculations.

## Directory Structure

All Rootstock state lives under a single root directory:

```
{root}/
├── environments/           # Environment files (*.py with PEP 723 metadata)
│   ├── mace_env.py        # Use _env suffix to avoid shadowing packages
│   └── chgnet_env.py
└── cache/
    └── huggingface/       # HuggingFace model weights (persisted)
```

## Running on Modal

```bash
# Initialize the volume (first time only)
modal run modal_app.py::init_rootstock_volume

# Test the v0.3 API with caching
modal run modal_app.py::test_new_api

# Run benchmarks
modal run modal_app.py::benchmark_v3

# Inspect cache contents
modal run modal_app.py::inspect_cache
```

## Local Development

```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Run linting
ruff check rootstock/
ruff format rootstock/
```

## Environment Files

Environment files are Python scripts with PEP 723 metadata and a `setup()` function:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///
"""MACE environment for Rootstock."""

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
```

## Performance

IPC overhead is <5% for systems with 1000+ atoms compared to direct in-process execution.

| System Size | Atoms | Typical Overhead |
|-------------|-------|------------------|
| Small       | 64    | ~10-15%          |
| Medium      | 256   | ~5-8%            |
| Large       | 1000  | <5%              |

## Legacy API (v0.2)

The v0.2 API with explicit `environment=` parameter still works:

```python
with RootstockCalculator(
    environment="/path/to/mace.py",
    model="medium",
    device="cuda",
) as calc:
    ...
```
