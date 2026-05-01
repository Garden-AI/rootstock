# API Reference

## RootstockCalculator

The main interface to Rootstock is the `RootstockCalculator` class, an ASE-compatible calculator.

### Basic Usage

```python
from ase.build import bulk
from rootstock import RootstockCalculator

atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

with RootstockCalculator(
    cluster="della",
    model="mace",
    checkpoint="medium",
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
| `cluster` | `str` | Yes* | Cluster name (e.g., `"della"`, `"sophia"`) |
| `root` | `str` | Yes* | Custom root path instead of a known cluster |
| `model` | `str` | Yes | MLIP family: `"mace"`, `"chgnet"`, `"uma"`, `"tensornet"` |
| `checkpoint` | `str` | No | Specific model weights (uses environment default if omitted) |
| `device` | `str` | No | `"cuda"` (default) or `"cpu"` |

*Either `cluster` or `root` must be provided, but not both.

### Examples

```python
# Using a known cluster with explicit checkpoint
RootstockCalculator(cluster="della", model="mace", checkpoint="medium")

# Using a known cluster with default checkpoint
RootstockCalculator(cluster="della", model="uma")

# Using a custom root path
RootstockCalculator(root="/scratch/gpfs/specific/install/path/rootstock", model="mace")
```

### Context Manager

`RootstockCalculator` should be used as a context manager to ensure proper cleanup of the worker subprocess:

```python
with RootstockCalculator(...) as calc:
    # Use the calculator
    atoms.calc = calc
    energy = atoms.get_potential_energy()
# Worker process is automatically terminated when exiting the context
```

!!! tip "Manual Cleanup"
    If you cannot use a context manager, call `calc.close()` manually to terminate the worker process.

## Available Models

Available models vary by cluster and change as new environments are added. See the [Example Configs](clusters.md) page for current deployments on each cluster.

### Model Reference

| Model | Environment | Default Checkpoint | Other Checkpoints |
|-------|-------------|-------------------|-------------------|
| `mace` | mace_env | `medium` | `small`, `large` |
| `chgnet` | chgnet_env | (pretrained) | — |
| `uma` | uma_env | `uma-s-1p1` | — |
| `tensornet` | tensornet_env | `TensorNet-MatPES-PBE-v2025.1-PES` | Other MatGL models |

### Checking Available Models

To see available models on your cluster, check the [Example Configs](clusters.md) page or run:

```bash
rootstock status
```

This command displays installed environments, their checkpoints, and cache sizes.

## CLI Reference

The Rootstock CLI provides commands for both administrators (setting up clusters) and users (querying available environments).

### User Commands

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

### Administrator Commands

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
rootstock new-env mace -o ./environments/mace_env.py

# Overwrite existing file
rootstock new-env mace --force
```

#### `rootstock install`

Build environment(s) from a file or directory. Builds the venv only — no model weights. Use `rootstock add` separately to download and verify checkpoints.

```bash
# Install from a single file
rootstock install ./mace_env.py

# Install all environments from a directory
rootstock install ./environments/

# Rebuild an existing environment
rootstock install mace_env --force

# Install without pushing manifest to backend
rootstock install mace_env.py --no-push
```

Options:

- `--root <path>`: Specify root directory (or use `$ROOTSTOCK_ROOT`)
- `--force`: Update registration and rebuild if environment exists
- `--verbose`, `-v`: Verbose output
- `--no-push`: Skip pushing manifest to backend

!!! note "`--models` was removed in v0.8.0"
    Pre-downloading weights at install time is now a separate step. Use `rootstock add <env> <checkpoint>` instead. Passing `--models` to `install` will exit with a migration error.

#### `rootstock add`

Download and verify a checkpoint for a built environment. Idempotent — safe to re-run.

```bash
# Login node (CPU, has network): download only
rootstock add mace medium --no-verify

# GPU node (no network): skip download (already fetched), verify on GPU
rootstock add mace medium

# Forward extra kwargs to setup() — values are JSON-decoded, fall back to strings
rootstock add uma uma-s-1p1 --kwarg task=omat
rootstock add some_env some_ckpt --kwarg charge=-1 --kwarg enabled=true
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
rootstock smoke-test --env mace --checkpoint medium

# JSON summary for cron
rootstock smoke-test --json
```

Exit code is 0 if all tested checkpoints passed, 1 otherwise.

!!! note "Smoke-test always uses default kwargs"
    `smoke-test` calls each env's `setup()` with no extra kwargs. A checkpoint that only works with non-default kwargs will appear failing here even though `add` succeeded — make the preferred kwargs the env's default if you need it to pass nightly.

#### `rootstock serve`

Start a worker process for an external i-PI server (advanced usage).

```bash
rootstock serve mace \
  --socket /tmp/ipi_socket \
  --checkpoint medium \
  --device cuda
```

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
