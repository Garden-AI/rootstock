# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rootstock runs MLIP (Machine Learning Interatomic Potential) calculators in isolated pre-built Python environments on HPC clusters, communicating via the i-PI protocol over Unix sockets.

Versioning is dynamic (git tags via uv-dynamic-versioning) — check `rootstock --version`. Manifest schema v6 (older schemas migrate in place on load; verification is per-cluster, and shared installs like sophia/polaris push one manifest per cluster); canonical-checkpoint-id API.

## Commands

### Local Development
```bash
uv sync   # creates .venv with the package (editable) + dev dependency group
```

### CLI Commands
```bash
# Build a pre-built environment (venv only — no model weights). First build
# writes environments/<name>.py.lock; rebuilds honor it unless --upgrade.
rootstock install <env_source.py> [--root <path>] [--force] [--upgrade]

# Download + verify a checkpoint by canonical id (idempotent). Use --no-verify on login nodes.
rootstock add <checkpoint-id> [--kwarg key=val ...] [--device cuda] [--no-verify]

# Re-verify all fetched checkpoints, plus each '<family>:custom' weights= path
# (re-loads a same-family checkpoint's cached weights and compares results).
# Suitable for nightly cron. --cluster names the machine (required on shared
# installs like sophia/polaris; add/sync take it too).
rootstock smoke-test [--env ENV] [--checkpoint CKPT] [--device cuda] [--cluster NAME] [--json]

# Show status (per-checkpoint verified/stale grid; --json for machine-readable)
rootstock status [--root <path>] [--json]

# Read-only permission check of install/cache roots + ancestors (exit 1 on issues)
rootstock check-perms [<root>] [--cluster <name>] [--group <group>] [--json]

# List environments
rootstock list --root <path>
```

### Linting, typechecking, tests
```bash
uv run ruff check rootstock/ tests/
uv run ruff format rootstock/ tests/
uv run ty check
uv run pytest --cov
```
CI (`.github/workflows/ci.yml`) runs the same four checks on every PR;
`ruff format` runs there with `--check`.

## Architecture

```
Main Process                          Worker Process (subprocess)
┌─────────────────────────┐          ┌─────────────────────────────┐
│ RootstockCalculator     │          │ Pre-built venv Python       │
│ (ASE-compatible)        │          │ (envs/mace/bin/python)      │
│                         │          │                             │
│ server.py (i-PI server) │◄────────►│ worker.py (i-PI client)     │
│ - sends positions       │   Unix   │ - receives positions        │
│ - receives forces       │  socket  │ - calculates forces         │
└─────────────────────────┘          └─────────────────────────────┘
```

**Key design**: pre-built virtual environments instead of dynamic `uv run`. This provides:
- Fast startup (no pip install at runtime)
- Works on any filesystem (no lock files or hardlinks needed)
- Reproducible environments

### Core Files

- `rootstock/cli.py` + `rootstock/commands/` - CLI (`install`, `add`, `smoke-test`, `status`, `serve`, ... — thin adapters over `rootstock/operations.py`)
- `rootstock/calculator.py` - ASE Calculator interface (main entry point)
- `rootstock/server.py` - Spawns worker subprocess, manages socket lifecycle
- `rootstock/worker.py` - i-PI client state machine
- `rootstock/environment.py` - Pre-built environment management, wrapper generation
- `rootstock/clusters.py` - Cluster registry and known environments

### Directory Structure

```
{root}/
├── layout.json             # on-disk layout version (clients refuse newer layouts)
├── .python/                # uv-managed Python interpreters (portable)
│   └── cpython-3.11.9-linux-x86_64-gnu/
├── environments/           # Environment SOURCE files (*.py with PEP 723 + CHECKPOINTS)
│   ├── mace.py
│   ├── mace.py.lock        # uv lockfile — rebuilds resolve from this
│   ├── uma.py
│   └── tensornet.py
├── envs/                   # Pre-built virtual environments
│   ├── mace/
│   │   ├── bin/python      # Symlinks to .python/
│   │   ├── lib/python3.11/site-packages/
│   │   ├── env_source.py   # Copy of source for imports
│   │   └── env_source.py.lock  # what this build was resolved from
│   └── uma/
└── cache/                  # XDG_CACHE_HOME for model weights
    ├── mace/
    └── huggingface/
```

The install root is self-contained and portable. Python interpreters are stored
in `.python/` so venv symlinks resolve correctly on any machine where the root
is mounted (HPC shared filesystem, NFS, Lustre, etc.).

On clusters where the right filesystem for code is different from the right
filesystem for the model-weight cache (e.g., Perlmutter — code on CFS, cache
on PSCRATCH), the install declares its own `cache_root` in `{root}/layout.json`
(written by `install`/`init`). The cluster registry is only a name → path
bootstrap; its `cache_root` field is a fallback for legacy installs that
predate the declaration. Most clusters use the same path for both.

### Known Clusters

| Cluster | Install Root | Cache Root (if split) |
|---------|--------------|-----------------------|
| `della` | `/scratch/gpfs/ROSENGROUP/common/rootstock` | (same as install root) |
| `sophia` | `/lus/eagle/projects/Garden-Ai/rootstock` | (same as install root) |
| `polaris` | `/lus/eagle/projects/Garden-Ai/rootstock` (shared with sophia) | (same as install root) |
| `perlmutter` | `/global/cfs/cdirs/m5268/rootstock` | `/pscratch/sd/o/oprice/rootstock-cache` |
| `delta` | `/work/hdd/data/rootstock` | (same as install root) |
| `frontier` | `/sw/frontier/ums/ums047/rootstock` | `/lustre/orion/ums047/world-shared/rootstock-cache` |

The sophia/polaris registry path is a *serving copy* under the Garden-Ai project, mirrored from the `/eagle/projects/Rootstock/rootstock` build root (not world-readable); serving is group-scoped (Garden-Ai project membership), and admin sync/smoke-test jobs target the build root via explicit `--root`/`ROOTSTOCK_ROOT`.

sophia/polaris share one install and one manifest (`clusters: ["sophia", "polaris"]`), but verification records and pushed manifests are per-cluster; verify-recording commands there need `--cluster`. An env source may declare `CLUSTERS = ["polaris"]` to ship a cluster-specific variant (see `docs/environments.md`); smoke-test selection and the pushed payloads are checkpoint-first — each id is tested and reported via the env it resolves to on that cluster.

## API

```python
# Single canonical checkpoint id; the hosting env is resolved automatically.
with RootstockCalculator(
    cluster="delta",
    checkpoint="mace-mp-0-medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()

# Forward extra kwargs to the env's setup() function. Cannot contain
# "checkpoint" or "device" — those are passed at the top level. Multi-head
# models (UMA `task`, MACE-MH-1 `head`) REQUIRE a head selection here — no
# default head; the env's VERIFY_KWARGS covers smoke-test/add instead.
with RootstockCalculator(
    cluster="delta",
    checkpoint="uma-s-1p1",
    setup_kwargs={"task": "omol"},
) as calc:
    ...

# atoms.info is forwarded to the worker every calculation (JSON-safe subset;
# numpy converted, non-serializable values dropped with a debug log), so
# OMol/POLAR checkpoints see charge/spin/external_field as if in-process.
# An info-only change invalidates the cached result (check_state override).
atoms.info["charge"] = 1

# User-supplied weights (e.g. a fine-tune): use the "<family>:custom"
# CHECKPOINTS entry for the model family (shown by `rootstock list`) and
# pass the weights file. The entry only selects the hosting env (no shipped
# weights are involved — its CHECKPOINTS value is None); the worker loads
# via the env's setup_from_path() instead of setup(). Requires weights=
# and vice versa.
with RootstockCalculator(
    cluster="delta",
    checkpoint="uma:custom",
    weights="/scratch/me/my-uma-ft.pt",
) as calc:
    ...
```

### Canonical checkpoint ids (bundled envs)

| Env | Canonical ids |
|---|---|
| `mace` | `mace-mp-0-{small,medium,large}`, `mace-off23-{small,medium,large}` |
| `esen` | `esen-md-direct-all-omol`, `esen-sm-conserving-all-omol`, `esen-sm-direct-all-omol` |
| `orb` | `orb-v2` |
| `tensornet` | `tensornet-matpes-pbe-2025-2` |
| `uma` | `uma-s-1p1`, `uma-s-1p2p1`, `uma-m-1p1` |

## Build Process

```bash
# 1. Create environment source file
cat > environments/mace.py << 'EOF'
# /// script
# requires-python = ">=3.11"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///

CHECKPOINTS = {
    "mace-mp-0-small":  "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large":  "large",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
EOF

# 2. Build pre-built environment (venv only — no model weights)
rootstock install environments/mace.py --root /path/to/rootstock

# 3. Download and verify checkpoints by canonical id (idempotent; use --no-verify on login nodes)
rootstock add mace-mp-0-medium --root /path/to/rootstock
rootstock add mace-mp-0-small --root /path/to/rootstock

# 4. Verify install state
rootstock status --root /path/to/rootstock
```
