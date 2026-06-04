# Architecture

This page explains how Rootstock runs a model.

![Rootstock runtime: an ASE calculator proxies to an isolated model in a managed subprocess on the same GPU node, exchanging positions and forces over a local Unix socket.](assets/rootstock_runtime.png)

## Overview

When you create a `RootstockCalculator`, Rootstock spawns a worker subprocess that runs the MLIP in its own pre-built virtual environment on the same node. Your own environment only needs the lightweight `rootstock` package, not the model's dependencies. The main process and worker communicate over a Unix domain socket using the [i-PI protocol](http://ipi-code.org/). All communication is local to one node; there are no remote network calls.

The worker loads the model once and keeps it warm, so repeated calls reuse the loaded weights.

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

Each MLIP runs in its own isolated environment with its exact dependencies. This removes version conflicts between models, including models that require incompatible Python or library versions. You can swap models in one line, change `checkpoint="mace-mp-0-medium"` to `checkpoint="uma-s-1p1"`, and use several models sequentially in the same script.

## Tradeoffs

Inter-process communication adds a small cost. On an 864-atom system the overhead is about 4%. Positions and forces are serialized with the i-PI protocol and pass over a local Unix domain socket. For most workloads this is negligible next to the MLIP forward pass.

## Directory Structure

After setup, the Rootstock root directory looks like this:

```
{root}/
├── .python/                # uv-managed Python interpreters
├── environments/           # Environment source files (*.py with PEP 723 metadata)
│   ├── mace.py
│   ├── uma.py
│   └── tensornet.py
├── envs/                   # Pre-built virtual environments
│   ├── mace/
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
