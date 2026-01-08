# Rootstock IPC Overhead Benchmark

## Goal

Determine whether inter-process communication (IPC) overhead is acceptable for Rootstock's architecture, where MLIP calculators run in isolated Python environments and communicate with the calling process via sockets.

**Success Criterion:** <5% overhead for systems with 1000+ atoms compared to direct in-process MLIP execution.

## Architecture Under Test

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  RootstockCalculator (ASE-compatible)               │    │
│  │  - Acts as i-PI server                              │    │
│  │  - Sends positions, receives forces                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                    i-PI protocol                             │
│                    (Unix socket)                             │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
┌───────────────────────────┼──────────────────────────────────┐
│              Worker Process (isolated env)                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  MACE Calculator                                    │    │
│  │  - Acts as i-PI client                              │    │
│  │  - Persistent across calculations                   │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Protocol

We use the i-PI protocol, a binary protocol designed for atomic simulations:
- 12-byte ASCII command strings
- Raw numpy array transmission (no JSON/msgpack serialization)
- Unix domain sockets for same-machine communication

This is the same protocol used by Quantum ESPRESSO, FHI-aims, VASP, and other codes for socket-based communication.

## Benchmarks

1. **Direct**: MACE-MP-0 running in the same process (baseline)
2. **IPC**: MACE-MP-0 running in subprocess, communicating via i-PI over Unix socket

### Test Matrix

| System Size | Atoms | Expected MACE Time | Max Acceptable Overhead |
|-------------|-------|-------------------|------------------------|
| Small       | 64    | ~10-20ms          | N/A (overhead may dominate) |
| Medium      | 256   | ~20-40ms          | ~5-10% |
| Large       | 1000  | ~50-100ms         | <5% |
| XL          | 4000  | ~200-500ms        | <2% |

## Running on Modal

```bash
# Install modal if needed
pip install modal

# Set up Modal (first time only)
modal setup

# Run the benchmark
modal run modal_app.py::benchmark_all
```

## Local Development

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run benchmark locally (requires GPU)
python -m benchmarks.run_local
```

## Project Structure

```
rootstock-ipc-benchmark/
├── README.md
├── pyproject.toml
├── modal_app.py              # Modal app definition and entry points
├── rootstock/
│   ├── __init__.py
│   ├── protocol.py           # i-PI protocol implementation
│   ├── server.py             # Socket server (main process)
│   ├── worker.py             # Socket client worker (subprocess)
│   └── calculator.py         # ASE-compatible calculator wrapper
└── benchmarks/
    ├── __init__.py
    ├── systems.py            # Test system generation (Cu supercells)
    ├── direct.py             # Direct MACE benchmark
    ├── ipc.py                # IPC MACE benchmark
    └── run_local.py          # Local benchmark runner
```

## Key Files to Review

- `rootstock/protocol.py`: Minimal i-PI protocol (adapted from ASE)
- `rootstock/worker.py`: The subprocess that runs MACE
- `benchmarks/ipc.py`: The IPC benchmark implementation

## References

- [i-PI Documentation](https://docs.ipi-code.org/)
- [ASE SocketIOCalculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html)
- [MACE](https://github.com/ACEsuit/mace)
