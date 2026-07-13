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

Commands are 12-byte ASCII strings; payloads are raw numpy buffers in
atomic units, following ASE's `ase.calculators.socketio` implementation.

### Deviations from standard i-PI

**Rootstock speaks a private dialect of the i-PI protocol.** The wire
framing is standard, but the semantics deviate in two load-bearing ways.
Read this before trying to point any other i-PI implementation at a
rootstock worker.

**The INIT payload is required, and it is JSON.** Standard i-PI transmits
only geometry (cell and positions); a force engine is expected to already
know its chemical species from its own input. A rootstock worker hosts a
generic ASE calculator and knows nothing about the system at spawn time, so
the server packs the species into the INIT message's free-form init string
as UTF-8 JSON:

```json
{"numbers": [1, 1, 8], "pbc": [true, true, true]}
```

A worker that has not received this payload fails at the first POSDATA.
Unknown keys in the JSON object are ignored, which makes new keys a
sanctioned forward-compatible extension channel.

**The worker demands INIT before every force evaluation.** After each
FORCEREADY the worker returns to NEEDINIT rather than READY, and the server
re-sends the JSON INIT every cycle. This is how composition and PBC changes
propagate mid-run (e.g. LAMMPS `fix gcmc` alongside `fix rootstock`).

Consequences:

- A standard i-PI server — i-PI itself, ASE's `SocketIOCalculator`, or any
  other implementation — **cannot drive a rootstock worker**: it won't send
  the JSON INIT payload, so the worker aborts at the first force request.
- The supported protocol peers are exactly `RootstockServer` (the ASE
  calculator path and `rootstock serve`'s counterpart) and
  `lammps/fix_rootstock.cpp` (which implements the dialect natively,
  including the per-cycle INIT).
- This is a deliberate, documented limitation of 1.0. The worker side of the
  protocol is frozen inside every built environment, so the requirement
  cannot be relaxed for environments that have already been deployed.

## Worker environment variables

The worker process reads a few `ROOTSTOCK_*` environment variables once at
startup, into a frozen config object (`rootstock/worker_config.py`).
Because every built env pins its own copy of rootstock, the worker code
inside an env cannot be updated without a rebuild — environment variables
set at spawn are the one configuration channel that always reaches
already-deployed workers, and `worker_config.py` is deliberately small,
stdlib-only, and never-fatal on bad input, since it freezes along with the
rest of the worker surface.

| Variable | Default | Description |
|---|---|---|
| `ROOTSTOCK_WORKER_CONNECT_RETRIES` | `50` | Connection attempts to the server socket |
| `ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY` | `0.1` | Seconds between connection attempts |
| `ROOTSTOCK_WORKER_LOG` | unset | Worker log destination when the client didn't attach one: `stderr`, `stdout`, or a file path (appended) |

Invalid values fall back to the defaults rather than killing the worker.
