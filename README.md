# Rootstock

A proof-of-concept package for running MLIP (Machine Learning Interatomic Potential) calculators in isolated pre-built Python environments, communicating via the i-PI protocol over Unix sockets. Currently deployed on Princeton's Della cluster.

## Quick Start

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
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
```

**Note:** Environments must be pre-built before use. See [Administrator Setup](#administrator-setup).

## Installation

```bash
pip install rootstock
# or
uv pip install rootstock
```

## API

The `model` parameter selects the environment family. The optional `checkpoint` parameter selects specific model weights (defaults to the environment's default if omitted).

```python
# Explicit checkpoint
RootstockCalculator(cluster="della", model="mace", checkpoint="medium")

# Default checkpoint (each environment has a sensible default)
RootstockCalculator(cluster="della", model="uma")

# Custom root path instead of a known cluster
RootstockCalculator(root="/scratch/gpfs/ROSENGROUP/common/rootstock", model="mace")
```

## Available Models

| Model | Checkpoint | Description |
|-------|------------|-------------|
| `mace` | `small` | MACE-MP-0 small |
| `mace` | `medium` (default) | MACE-MP-0 medium |
| `chgnet` | _(pretrained)_ | CHGNet pretrained universal model |
| `uma` | `uma-s-1p1` (default) | Meta UMA small model (FAIRChem) |
| `tensornet` | `TensorNet-MatPES-PBE-v2025.1-PES` (default) | TensorNet MatPES PBE (MatGL) |

## Architecture

```
Main Process                          Worker Process (subprocess)
+-------------------------+          +-----------------------------+
| RootstockCalculator     |          | Pre-built venv Python       |
| (ASE-compatible)        |          | (mace_env/bin/python)       |
|                         |          |                             |
| server.py (i-PI server) |<-------->| worker.py (i-PI client)     |
| - sends positions       |   Unix   | - receives positions        |
| - receives forces       |  socket  | - calculates forces         |
+-------------------------+          +-----------------------------+
```

The worker process uses a pre-built virtual environment, providing:
- **Fast startup**: No dependency installation at runtime
- **Filesystem compatibility**: Works on NFS, Lustre, GPFS
- **Reproducibility**: Same environment every time

## Directory Structure

```
{root}/
├── .python/                # uv-managed Python interpreters (portable)
├── environments/           # Environment SOURCE files (*.py with PEP 723)
│   ├── mace_env.py
│   ├── chgnet_env.py
│   ├── uma_env.py
│   └── tensornet_env.py
├── envs/                   # Pre-built virtual environments
│   ├── mace_env/
│   │   ├── bin/python
│   │   ├── lib/python3.11/site-packages/
│   │   └── env_source.py   # Copy of environment source
│   └── ...
├── home/                   # Fake HOME for build & workers
│   ├── .cache/fairchem/    # FAIRChem weights
│   └── .matgl/             # MatGL weights
└── cache/                  # XDG_CACHE_HOME for well-behaved libs
    ├── mace/
    └── huggingface/
```

## CLI Commands

```bash
# Build a pre-built environment
rootstock build <env_name> --root <path> [--models m1,m2] [--force]

# Show status
rootstock status --root <path>

# Register an environment source file
rootstock register <env_file> --root <path>

# List environments
rootstock list --root <path>
```

## Administrator Setup

Environments must be pre-built before users can run calculations.

### 1. Create Directory Structure

```bash
mkdir -p /scratch/gpfs/ROSENGROUP/common/rootstock/{environments,envs,cache,home}
```

### 2. Register Environment Source Files

```bash
rootstock register environments/mace_env.py --root /scratch/gpfs/ROSENGROUP/common/rootstock
rootstock register environments/chgnet_env.py --root /scratch/gpfs/ROSENGROUP/common/rootstock
rootstock register environments/uma_env.py --root /scratch/gpfs/ROSENGROUP/common/rootstock
rootstock register environments/tensornet_env.py --root /scratch/gpfs/ROSENGROUP/common/rootstock
```

### 3. Build Environments

```bash
ROOT=/scratch/gpfs/ROSENGROUP/common/rootstock

rootstock build mace_env --root $ROOT --models small,medium
rootstock build chgnet_env --root $ROOT
rootstock build uma_env --root $ROOT --models uma-s-1p1
rootstock build tensornet_env --root $ROOT

# Verify
rootstock status --root $ROOT
```

## Local Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check rootstock/
ruff format rootstock/
```

## Experimental: LAMMPS Support

Rootstock can drive LAMMPS simulations through a native `fix` that auto-spawns
a worker subprocess over Unix sockets using the i-PI protocol. No separate
process management needed — add one line to your input script:

```
fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu
```

### Building

The fix ships as two files (`fix_rootstock.h`, `fix_rootstock.cpp`) with no
dependencies beyond the C++ standard library and POSIX sockets. Copy them into
your LAMMPS `src/` directory and rebuild:

```bash
./lammps/install.sh /path/to/lammps/src
cd /path/to/lammps/build
cmake ../cmake [your usual flags]
make -j 4
```

Rootstock must also be installed and on `PATH` so the fix can call
`rootstock resolve` and `rootstock serve`:

```bash
pip install rootstock
```

### Syntax

```
fix <id> <group> rootstock cluster <n> model <model> \
    checkpoint <ckpt> device <dev> [timeout <sec>] elements <e1> <e2> ...
```

| Keyword | Required | Default | Description |
|---------|----------|---------|-------------|
| `cluster` | yes | — | Cluster name (e.g., `della`) |
| `model` | yes | — | `mace`, `chgnet`, `uma`, `tensornet` |
| `checkpoint` | no | `default` | Model weights (e.g., `medium`, `uma-s-1p1`) |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `timeout` | no | `120` | Seconds to wait for worker startup |
| `elements` | yes | — | Element symbols mapping atom types (must be last) |

### Example: NPT

```
units metal
boundary p p p
read_data structure.data
mass 1 63.546
mass 2 16.000

pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock cluster della model uma checkpoint uma-s-1p1 device cuda elements Cu O

velocity all create 300 300
fix 1 all npt temp 300 300 0.1 iso 0 0 1.0
timestep 0.001

thermo_style custom step temp press vol f_mlip
thermo 100
run 10000
```

The fix contributes virial information, so barostats (`npt`, `nph`) work
correctly. Energy is accessible via `f_mlip` in thermo output.

### Notes

- Requires `units metal`. The fix checks this at startup.
- Use `pair_style zero` unless you intentionally want to combine potentials
  (the fix adds forces via `+=`).
- Single-node only — the worker sees all atoms and computes its own
  neighborhoods.
