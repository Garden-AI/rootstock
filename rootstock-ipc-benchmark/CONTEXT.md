# Claude Code Context

This file provides context for Claude Code to understand the project goals, design decisions, and areas that may need work.

## Project Goal

Validate whether IPC overhead is acceptable for Rootstock's architecture. Rootstock will provide isolated Python environments for MLIP calculators on HPC clusters, so researchers don't need to manage complex MLIP dependencies themselves.

**Success Criterion:** <5% overhead for systems with 1000+ atoms.

## Design Decisions

### Why i-PI Protocol?
- Battle-tested: used by QE, FHI-aims, VASP, LAMMPS, etc.
- Binary numpy arrays = minimal serialization overhead
- ASE already has support for it
- Unix domain sockets for same-machine = fast

### Architecture
```
Main Process (user code)          Worker Process (isolated env)
┌─────────────────────────┐      ┌─────────────────────────────┐
│ RootstockCalculator     │      │ MACE Calculator             │
│ (ASE-compatible)        │      │ (loaded once, persistent)   │
│                         │      │                             │
│ Acts as i-PI SERVER     │◄────►│ Acts as i-PI CLIENT         │
│ Sends positions         │      │ Receives positions          │
│ Receives forces         │      │ Calculates & sends forces   │
└─────────────────────────┘      └─────────────────────────────┘
         Unix domain socket
```

### Key Files
- `rootstock/protocol.py` - Minimal i-PI protocol implementation (adapted from ASE)
- `rootstock/server.py` - Server that spawns and communicates with worker
- `rootstock/worker.py` - Worker process that runs MACE
- `rootstock/calculator.py` - ASE-compatible calculator wrapper
- `modal_app.py` - Modal app for running on GPU

## Known Issues / TODOs

1. **Worker currently hardcoded to Cu atoms** - The i-PI protocol doesn't have a clean way to send atomic numbers. For the benchmark this is fine since we're testing Cu systems, but for production we'd need to extend the protocol (perhaps via the INIT message's extra bytes).

2. **Error handling is minimal** - Per user request, error handling is minimal for this PoC. Production code would need proper timeout handling, worker restart logic, etc.

3. **Environment isolation not yet implemented** - The worker currently uses the same Python environment. For true Rootstock, it would use a separate uv-managed environment. This doesn't affect the IPC overhead measurement.

4. **MACE model loading** - The worker uses `mace_mp()` which downloads models. For benchmarking, ensure the model is cached first.

## Running the Benchmarks

### On Modal (recommended)
```bash
pip install modal
modal setup  # First time only
modal run modal_app.py::benchmark_all
```

### Locally (requires GPU + MACE)
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[mace]"
python -m benchmarks.run_local
```

### Quick IPC overhead test (no MLIP)
```bash
modal run modal_app.py::benchmark_ipc_overhead
# or
python -m benchmarks.run_local --ipc-only
```

## Expected Results

Based on protocol analysis:
- **Pure IPC overhead**: ~0.1-0.5ms per call
- **1000-atom MACE calculation**: ~50-100ms on GPU
- **Expected overhead percentage**: <1-2%

If overhead is higher than expected, potential causes:
1. Socket buffer sizing
2. Python GIL contention
3. Serialization overhead (shouldn't be - we use raw bytes)
4. GPU synchronization timing

## Contact

This is a proof-of-concept for the Rootstock project. The goal is to share benchmark results with Andrew Rosen (QuAcc creator, ASE maintainer) to validate the approach.
