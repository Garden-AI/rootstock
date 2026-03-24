# Architecture

## Overview

When you create a `RootstockCalculator`, Rootstock spawns a subprocess that runs the MLIP in its own pre-built virtual environment. The main process and worker communicate over a Unix domain socket using the [i-PI protocol](http://ipi-code.org/). All communication occurs on a single node with no remote network calls.

```
Your script (on cluster node)          Worker subprocess
+-------------------------+          +-----------------------------+
| RootstockCalculator     |          | Pre-built MLIP environment  |
| (ASE-compatible)        |          |                             |
|                         |          |                             |
| server.py (i-PI server) |<-------->| worker.py (i-PI client)     |
| - sends positions       |   Unix   | - receives positions        |
| - receives forces       |  socket  | - calculates forces         |
+-------------------------+          +-----------------------------+
```

## Design Benefits

This design eliminates environment conflicts when experimenting with different MLIPs or using multiple MLIPs in a single workflow:

- **No environment conflicts**: Each MLIP runs in isolation with its exact required dependencies
- **One-line model swapping**: Change `model="mace"` to `model="uma"` without reinstalling anything
- **Multi-model workflows**: Use multiple MLIPs in the same script (sequentially)
- **Clean user environments**: Users only install the lightweight `rootstock` package

## Tradeoffs

The architecture introduces minimal overhead due to inter-process communication:

- **~4% overhead** on an 864-atom system
- Communication occurs via Unix domain socket (fast, local-only)
- Positions and forces are serialized using the i-PI protocol

For most use cases, this overhead is negligible compared to MLIP forward pass time.

## Directory Structure

After setup, the Rootstock root directory looks like this:

```
{root}/
├── .python/                # uv-managed Python interpreters
├── environments/           # Environment source files (*.py with PEP 723 metadata)
│   ├── mace_env.py
│   ├── chgnet_env.py
│   ├── uma_env.py
│   └── tensornet_env.py
├── envs/                   # Pre-built virtual environments
│   ├── mace_env/
│   │   ├── bin/python
│   │   ├── lib/python3.11/site-packages/
│   │   └── env_source.py
│   └── ...
├── home/                   # Redirected HOME for not-well-behaved libraries
│   ├── .cache/fairchem/
│   └── .matgl/
└── cache/                  # XDG_CACHE_HOME and HF_HOME for well-behaved libraries
    ├── mace/
    └── huggingface/
```

### Why the `home/` Directory?

Some ML libraries (FAIRChem, MatGL) ignore `XDG_CACHE_HOME` and write to `~/.cache/` unconditionally. Rootstock redirects `HOME` during environment builds and worker runtime to ensure model weights are stored in the shared directory rather than in individual user home directories.

## i-PI Protocol

Rootstock uses the [i-PI protocol](http://ipi-code.org/) for communication between the main process and worker:

1. Main process sends atomic positions and cell parameters
2. Worker receives positions and runs the MLIP forward pass
3. Worker sends back energy, forces, and stress
4. Main process receives results and returns them to ASE

The protocol is text-based and designed for interoperability between simulation codes.
