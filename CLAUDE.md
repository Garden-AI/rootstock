# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rootstock is a proof-of-concept for running MLIP (Machine Learning Interatomic Potential) calculators in isolated Python environments, communicating via the i-PI protocol over Unix sockets. This benchmark validates that IPC overhead is acceptable (<5% for 1000+ atom systems).

**Current version: v0.3** - Environment caching and simplified API.

## Commands

### Running on Modal
```bash
# Initialize the volume (first time only)
modal run modal_app.py::init_rootstock_volume

# Test v0.3 API with caching
modal run modal_app.py::test_new_api

# Run benchmarks
modal run modal_app.py::benchmark_v3

# Inspect cache contents
modal run modal_app.py::inspect_cache
```

### Local Development
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
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
│ RootstockCalculator     │          │ MLIPWorker                  │
│ (ASE-compatible)        │          │ (loads MACE once)           │
│                         │          │                             │
│ server.py (i-PI server) │◄────────►│ worker.py (i-PI client)     │
│ - sends positions       │   Unix   │ - receives positions        │
│ - receives forces       │  socket  │ - calculates forces         │
└─────────────────────────┘          └─────────────────────────────┘
```

**Key design**: The worker loads the MLIP model once and persists across calculations, avoiding model loading overhead per call.

### Core Files

- `rootstock/clusters.py` - Cluster registry and model string parsing
- `rootstock/calculator.py` - ASE Calculator interface (main entry point)
- `rootstock/server.py` - Spawns worker subprocess, manages socket lifecycle
- `rootstock/worker.py` - i-PI client state machine (NEEDINIT → READY → HAVEDATA → loop)
- `rootstock/protocol.py` - i-PI binary protocol (12-byte commands, raw numpy arrays)
- `rootstock/environment.py` - Wrapper script generation and spawn commands

### Directory Structure

```
{root}/
├── environments/           # Environment files (*.py with PEP 723 metadata)
│   ├── mace_env.py        # Use _env suffix to avoid shadowing packages
│   └── chgnet_env.py
└── cache/
    └── huggingface/       # HuggingFace model weights (persisted)
```

### Known Clusters

| Cluster | Root Path |
|---------|-----------|
| `modal` | `/vol/rootstock` |
| `della` | `/scratch/gpfs/SHARED/rootstock` |

## v0.3 API

```python
# Preferred: cluster + model
with RootstockCalculator(
    cluster="modal",
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

# Power user: direct environment path (legacy)
with RootstockCalculator(
    environment="/custom/path/mace.py",
    model="medium",
    device="cuda",
) as calc:
    ...
```

### Model String Format

- `"mace-medium"` → environment=mace_env.py, model_arg="medium"
- `"chgnet"` → environment=chgnet_env.py, model_arg=""
- `"mace-/path/to/weights.pt"` → environment=mace_env.py, model_arg="/path/to/weights.pt"
