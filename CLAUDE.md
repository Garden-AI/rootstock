# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rootstock is a proof-of-concept for running MLIP (Machine Learning Interatomic Potential) calculators in isolated Python environments, communicating via the i-PI protocol over Unix sockets. This benchmark validates that IPC overhead is acceptable (<5% for 1000+ atom systems).

## Commands

### Running Benchmarks on Modal (Recommended)
```bash
cd rootstock-ipc-benchmark
modal run modal_app.py::benchmark_all           # Full benchmark suite
modal run modal_app.py::benchmark_size --size large  # Single size
modal run modal_app.py::benchmark_ipc_overhead  # Pure IPC overhead test
```

### Local Development
```bash
cd rootstock-ipc-benchmark
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"           # Core + dev tools
uv pip install -e ".[mace]"          # Add MACE (requires GPU)
python -m benchmarks.run_local       # Run local benchmark
python -m benchmarks.run_local --ipc-only  # IPC overhead only
```

### Linting
```bash
ruff check rootstock-ipc-benchmark/
ruff format rootstock-ipc-benchmark/
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

- `rootstock/protocol.py` - i-PI binary protocol (12-byte commands, raw numpy arrays, atomic unit conversions)
- `rootstock/server.py` - Spawns worker subprocess, manages socket lifecycle
- `rootstock/worker.py` - i-PI client state machine (NEEDINIT → READY → HAVEDATA → loop)
- `rootstock/calculator.py` - ASE Calculator interface wrapping the server

### Protocol Flow

1. Server sends `STATUS` → Worker responds `READY`
2. Server sends `POSDATA` (cell + positions in Bohr)
3. Server sends `STATUS` → Worker responds `HAVEDATA`
4. Server sends `GETFORCE` → Worker responds `FORCEREADY` (energy, forces, virial)

## Known Limitations

- Worker hardcodes Cu atoms (i-PI protocol lacks clean atomic number transmission)
- Minimal error handling (intentional for PoC)
- Uses same Python environment (true isolation via uv not yet implemented)
