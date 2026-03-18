# Architecture

## Overview

When you create a `RootstockCalculator`, Rootstock spawns a subprocess that runs the MLIP in its own pre-built virtual environment. The main process and worker communicate over a Unix domain socket using the [i-PI protocol](http://ipi-code.org/). This happens on a single node (no remote network calls).

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

This design takes out the pain of environment conflicts when experimenting with different MLIPs or using multiple MLIPs in a single workflow:

- **No environment conflicts**: Each MLIP runs in isolation with its exact required dependencies
- **One-line model swapping**: Change `model="mace"` to `model="uma"` without reinstalling anything
- **Multi-model workflows**: Use multiple MLIPs in the same script (sequentially)
- **Clean user environments**: Users only install the lightweight `rootstock` package

## Tradeoffs

The tradeoff is that the architecture adds some overhead due to inter-process communication:

- **~4% overhead** on an 864 atom system
- Communication happens via Unix domain socket (fast, no network)
- Positions and forces are serialized using the i-PI protocol

For most use cases, this overhead is negligible compared to the time spent in the MLIP forward pass.

## Directory Structure

After setup, the rootstock root directory looks like this:

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

The `home/` directory exists because some ML libraries (FAIRChem, MatGL) ignore `XDG_CACHE_HOME` and write to `~/.cache/` unconditionally. Rootstock redirects `HOME` during builds and at worker runtime so that model weights end up in the shared directory rather than in individual users' home directories.

## i-PI Protocol

Rootstock uses the [i-PI protocol](http://ipi-code.org/) for communication between the main process and worker:

1. **Main process** sends atomic positions and cell parameters
2. **Worker** receives positions, runs the MLIP forward pass
3. **Worker** sends back energy, forces, and stress
4. **Main process** receives results and returns them to ASE

The protocol is text-based and designed for interoperability between different simulation codes.
