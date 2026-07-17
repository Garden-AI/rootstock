# API Reference

## RootstockCalculator

The main interface to Rootstock is the `RootstockCalculator` class, an ASE-compatible calculator.

### Basic usage

```python
from ase.build import bulk
from rootstock import RootstockCalculator

atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

with RootstockCalculator(
    cluster="della",
    checkpoint="mace-mp-0-medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `checkpoint` | `str` | Yes | Canonical checkpoint id (e.g., `"mace-mp-0-medium"`, `"uma-s-1p1"`). The hosting env is resolved automatically by walking the installed envs and matching against each env's `CHECKPOINTS` table |
| `cluster` | `str` | Yes* | Cluster name (e.g., `"della"`, `"perlmutter"`) |
| `root` | `str` | Yes* | Custom install-root path instead of a known cluster |
| `cache_root` | `str` | No | Override path for the model-weight cache and redirected `HOME`. When omitted, the install's own declaration (`{root}/layout.json`) decides, falling back to the cluster registry for legacy roots, then to `root` |
| `device` | `str` | No | `"cuda"` (default) or `"cpu"` |
| `setup_kwargs` | `dict` | No | Extra keyword arguments forwarded to the env's `setup()` function (e.g., `{"task": "omol"}`). Cannot contain `checkpoint` or `device` |
| `timeout` | `float` | No | Socket timeout in seconds for worker operations (default 600, matching checkpoint verification — so the first real force call, which may pay for `torch.compile` or large neighbor lists, runs under the envelope verification exercised) |

*`cluster` and `root` are mutually exclusive. When neither is given, the calculator falls back to the `ROOTSTOCK_ROOT` environment variable and then the `root` in `~/.config/rootstock/config.toml` — the same resolution the CLI uses — so on a configured machine `RootstockCalculator(checkpoint=...)` alone works.

### Examples

```python
# Using a known cluster
RootstockCalculator(cluster="della", checkpoint="mace-mp-0-medium")

# Perlmutter — cluster name bootstraps the install path; the install itself
# declares where its cache lives (layout.json), e.g. PSCRATCH
RootstockCalculator(cluster="perlmutter", checkpoint="uma-s-1p1")

# Custom install root (cache_root from the install's declaration, else the root)
RootstockCalculator(root="/scratch/gpfs/specific/install/rootstock", checkpoint="mace-mp-0-medium")

# Explicit split between install root and cache root
RootstockCalculator(
    root="/global/cfs/cdirs/myproj/rootstock",
    cache_root="/pscratch/sd/u/me/rootstock-cache",
    checkpoint="uma-s-1p1",
    setup_kwargs={"task": "omol"},
)
```

### Context manager

`RootstockCalculator` should be used as a context manager to ensure proper cleanup of the worker subprocess:

```python
with RootstockCalculator(...) as calc:
    # Use the calculator
    atoms.calc = calc
    energy = atoms.get_potential_energy()
# Worker process is automatically terminated when exiting the context
```

### Worker crashes and recovery

A worker that dies mid-calculation (GPU OOM, batch-system kill) raises `rootstock.WorkerDiedError` carrying a post-mortem: the process exit code and the tail of the worker's captured output. The calculator tears the dead server down, so the **same calculator instance recovers on the next call** — a fresh worker is started automatically. There is no automatic retry of the failed calculation: the same configuration would likely fail the same way, so retrying is the caller's decision.

```python
from rootstock import WorkerDiedError

try:
    energy = atoms.get_potential_energy()
except WorkerDiedError as exc:
    print(exc)          # exit code + worker stderr tail (e.g. the OOM traceback)
    ...                 # decide: smaller system, different device, give up
```

### Logging

Client-side diagnostics use stdlib logging under the `rootstock` namespace — server lifecycle on `rootstock.server` (INFO/DEBUG), the full i-PI wire trace on `rootstock.protocol` (DEBUG):

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # or logging.getLogger("rootstock").setLevel(...)
```

The worker subprocess is separate: its verbosity is controlled by the `ROOTSTOCK_WORKER_LOG` environment variable (`stderr`, `stdout`, or a file path), and its output is captured and shown automatically when the worker fails.

## Available models

What is deployed and verified per cluster changes over time. The authoritative, current list lives in the [Matter Model Almanac](https://garden-ai.github.io/almanac).

### Checkpoint reference

The table below lists the canonical checkpoint ids defined by the bundled env files. It illustrates the ids these envs expose; it is not a statement of what is installed on any given cluster. For current availability per cluster, consult the [Matter Model Almanac](https://garden-ai.github.io/almanac).

| Env | Canonical checkpoint ids |
|---|---|
| `mace` | `mace-mp-0-{small,medium,large}`, `mace-off23-{small,medium,large}` |
| `esen` | `esen-md-direct-all-omol`, `esen-sm-conserving-all-omol`, `esen-sm-direct-all-omol` |
| `orb` | `orb-v2` |
| `tensornet` | `tensornet-matpes-pbe-2025-2` |
| `uma` | `uma-s-1p1` |

### Checking what is installed

To see what is installed on your cluster, consult the [Matter Model Almanac](https://garden-ai.github.io/almanac), or run:

```bash
rootstock status
```

This command displays installed environments, their checkpoints, and cache sizes.

## CLI reference

The Rootstock CLI provides commands for both administrators (setting up clusters) and users (querying available environments).

### User commands

Commands users can run to explore available environments:

#### `rootstock status`

Display installation status, including all installed environments and cache usage.

```bash
rootstock status
```

#### `rootstock list`

List all registered environments in the shared environments directory.

```bash
rootstock list
```

#### `rootstock resolve`

Look up the root directory for a known cluster.

```bash
# Human-readable output
rootstock resolve --cluster della

# JSON output
rootstock resolve --cluster della --json
```

### Administrator commands

Commands for cluster administrators to set up and manage Rootstock installations:

#### `rootstock init`

Interactive setup wizard for creating a new Rootstock installation. Prompts for:

- Root directory path
- API credentials for dashboard integration (optional)
- Maintainer information

```bash
rootstock init

# Skip directory creation
rootstock init --skip-dirs

# Skip manifest initialization
rootstock init --skip-manifest
```

#### `rootstock new-env`

Create a new environment template file with the required PEP 723 structure.

```bash
# Create template in current directory
rootstock new-env mace

# Specify output path
rootstock new-env mace -o ./environments/mace.py

# Overwrite existing file
rootstock new-env mace --force
```

#### `rootstock install`

Build environment(s) from a file or directory. Builds the venv only — no model weights. Use `rootstock add` separately to download and verify checkpoints.

```bash
# Install from a single file
rootstock install ./mace.py

# Install all environments from a directory
rootstock install ./environments/

# Rebuild an existing environment (honors the env's lockfile)
rootstock install mace --force

# Rebuild and re-resolve dependencies to the latest allowed versions
rootstock install mace --force --upgrade

# Install without pushing manifest to backend
rootstock install mace.py --no-push
```

The first build resolves the env file's version ranges and writes a uv lockfile (`environments/<name>.py.lock`); later rebuilds install exactly the locked versions unless `--upgrade` is passed. See [Lockfiles and reproducible rebuilds](environments.md#lockfiles-and-reproducible-rebuilds).

Options:

- `--root <path>`: Specify root directory (or use `$ROOTSTOCK_ROOT`)
- `--force`: Update registration and rebuild if environment exists
- `--upgrade`: Re-resolve dependencies to the latest allowed versions instead of honoring the lockfile
- `--verbose`, `-v`: Verbose output
- `--no-push`: Skip pushing manifest to backend

#### `rootstock add`

Download and verify a checkpoint by canonical id. The hosting env is resolved by walking the installed envs and matching the id against each env's `CHECKPOINTS` table. Idempotent — safe to re-run.

```bash
# Login node (CPU, has network): download only
rootstock add mace-mp-0-medium --no-verify

# GPU node (no network): skip download (already fetched), verify on GPU
rootstock add mace-mp-0-medium

# Forward extra kwargs to setup() — values are JSON-decoded, fall back to strings
rootstock add uma-s-1p1 --kwarg task=omat
rootstock add esen-md-direct-all-omol --kwarg charge=-1 --kwarg enabled=true
```

Options:

- `--device <dev>`: Device for verification (default: `cuda`)
- `--no-verify`: Skip the verify phase (login-node escape hatch)
- `--kwarg KEY=VAL`: Repeatable extra kwarg passed to `setup()`. Values are JSON-decoded first; on parse failure, fall back to a string
- `--root <path>`: Root directory
- `--no-push`: Skip pushing manifest to backend

#### `rootstock smoke-test`

Re-verify checkpoints already in the manifest. Never downloads. Suitable for nightly cron.

```bash
# Test all fetched checkpoints
rootstock smoke-test

# Filter
rootstock smoke-test --env mace
rootstock smoke-test --env mace --checkpoint mace-mp-0-medium

# JSON summary for cron
rootstock smoke-test --json
```

Exit code is 0 if all tested checkpoints passed, 1 otherwise.

!!! note "Smoke-test always uses default kwargs"
    `smoke-test` calls each env's `setup()` with no extra kwargs. A checkpoint that only works with non-default kwargs will appear failing here even though `add` succeeded — make the preferred kwargs the env's default if you need it to pass nightly.

#### `rootstock serve`

Start a worker process for an external i-PI server (advanced usage). Takes a single canonical checkpoint id; the hosting env is resolved from the id.

```bash
# Create the socket inside a private directory — a socket directly in /tmp
# is world-visible and race-able by other users on shared nodes.
SOCKET_DIR=$(mktemp -d)
rootstock serve mace-mp-0-medium \
  --socket "$SOCKET_DIR/ipi.sock" \
  --device cuda
```

Options:

- `--socket <path>`: Unix socket path for the i-PI server. Place it inside a
  private (0700) directory, e.g. from `mktemp -d` — the LAMMPS styles and
  `RootstockServer` do this automatically for the sockets they create.
- `--device <dev>`: Device (default: `cuda`)
- `--kwarg KEY=VAL`: Repeatable extra kwarg passed to `setup()` (same JSON-decoding as `add`)

#### `rootstock manifest`

Manage the installation manifest that tracks environment state.

```bash
# Show current manifest
rootstock manifest show
rootstock manifest show --json

# Push manifest to dashboard
rootstock manifest push

# Initialize new manifest
rootstock manifest init --cluster della
rootstock manifest init --cluster della --force
```
