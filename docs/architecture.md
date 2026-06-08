# How does a RootstockCalculator work?

This page explains how Rootstock runs a model at runtime, from the `RootstockCalculator` in your script down to the worker subprocess.

![Rootstock runtime: an ASE calculator proxies to an isolated model in a managed subprocess on the same GPU node, exchanging positions and forces over a local Unix socket.](assets/rootstock_runtime.png)

## Overview

When you create a `RootstockCalculator`, Rootstock spawns a worker subprocess that runs the MLIP in its own pre-built virtual environment on the same node. Your own environment only needs the lightweight `rootstock` package, not the model's dependencies. The two communicate over a Unix domain socket using the [i-PI protocol](http://ipi-code.org/): the i-PI *server* (`server.py`) runs inside your process and drives the exchange, while the worker subprocess (`worker.py`) is the i-PI *client* that runs the forward pass. All communication is local to one node; there are no remote network calls.

The worker loads the model once and keeps it warm, so repeated calls reuse the loaded weights.

## Design benefits

Each MLIP runs in its own isolated environment with its exact dependencies. This removes version conflicts between models, including models that require incompatible Python or library versions. You can swap models in one line, change `checkpoint="mace-mp-0-medium"` to `checkpoint="uma-s-1p1"`, and use several models sequentially in the same script.

## Tradeoffs

Inter-process communication adds a small cost. On an 864-atom system the overhead is about 4%. Positions and forces are serialized with the i-PI protocol and pass over a local Unix domain socket. For most workloads this is negligible next to the MLIP forward pass.

## i-PI protocol

Rootstock uses the [i-PI protocol](http://ipi-code.org/) for communication between the main process and worker:

1. Main process sends atomic positions and cell parameters
2. Worker receives positions and runs the MLIP forward pass
3. Worker sends back energy, forces, and stress
4. Main process receives results and returns them to ASE

The protocol is text-based and designed for interoperability between simulation codes.
