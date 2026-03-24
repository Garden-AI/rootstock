# Rootstock

Rootstock makes it easy to use machine-learned interatomic potentials (MLIPs) on national lab and academic HPC clusters. Researchers can use multiple MLIPs (MACE, CHGNet, UMA, TensorNet, and others) with ASE or LAMMPS without managing the conflicting Python environments that each MLIP requires.

Rootstock provides an [ASE](https://wiki.fysik.dtu.dk/ase/)-compatible calculator that runs each MLIP in an isolated, pre-built Python environment behind the scenes. Swapping models is a one-line change, even if the MLIPs require different Python or library versions. Rootstock also integrates with [LAMMPS](https://www.lammps.org/) through a `fix` with any supported MLIP.

## Status

Rootstock is **early-stage software under active development.** It is currently deployed on two HPC clusters:

- **Della** — Princeton Research Computing
- **Sophia** — Argonne Leadership Computing Facility (ALCF)

We are looking for additional clusters and early users to help shape the tool. If you're interested in trying Rootstock on your cluster or for a specific project, please reach out to Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).

## Quick Start

Rootstock is designed for use on an HPC cluster where it has already been set up by a maintainer. The code below runs in your normal Python environment — inside a SLURM job script, an interactive session, or a Jupyter notebook on the cluster. Rootstock handles the MLIP environment isolation.

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

Changing `model="mace"` to `model="uma"` or `model="tensornet"` swaps the underlying potential.

## Installation

Users install only the lightweight `rootstock` package. The heavy ML dependencies (PyTorch, MACE, FAIRChem, etc.) live in the pre-built environments on the cluster.

```bash
pip install rootstock
# or
uv pip install rootstock
```

## API

The `model` parameter selects the MLIP family. The optional `checkpoint` parameter selects specific model weights (defaults to each environment's default if omitted).

```python
# Explicit checkpoint
RootstockCalculator(cluster="della", model="mace", checkpoint="medium")

# Default checkpoint (each environment defines a sensible default)
RootstockCalculator(cluster="della", model="uma")

# Custom root path instead of a known cluster
RootstockCalculator(root="/scratch/gpfs/specific/install/path/rootstock", model="mace")
```

## Available Models

The set of available models varies by cluster and changes as new environments are added. See the [example configs](https://garden-ai.github.io/rootstock/clusters/) in the docs for what is currently deployed on each cluster.

## Architecture

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

This design takes out the pain of environment conflicts when experimenting with different MLIPs or using multiple MLIPs in a single workflow. The tradeoff is that it adds some overhead due to the inter-process communication, around 4% on an 864 atom system.

## LAMMPS Support (Experimental)

Rootstock includes a native LAMMPS `fix` that auto-spawns a worker subprocess, giving LAMMPS users access to any Rootstock-managed MLIP. 

Add one line to your LAMMPS input script:

```
fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu
```

The fix handles worker lifecycle, socket communication, and cleanup automatically. Virial information is passed through, so barostats (`npt`, `nph`) work correctly. Energy is accessible via `f_mlip` in thermo output.

### Building the Fix

The fix ships as two files (`fix_rootstock.h`, `fix_rootstock.cpp`) with no dependencies beyond the C++ standard library and POSIX sockets. Copy them into your LAMMPS `src/` directory and rebuild:

```bash
./lammps/install.sh /path/to/lammps/src
cd /path/to/lammps/build
cmake ../cmake [your usual flags]
make -j 4
```

Rootstock must also be installed and on `PATH` so the fix can call `rootstock resolve` and `rootstock serve`:

```bash
pip install rootstock
```

### LAMMPS Fix Syntax

```
fix <id> <group> rootstock cluster <n> model <model> \
    checkpoint <ckpt> device <dev> [timeout <sec>] elements <e1> <e2> ...
```

| Keyword | Required | Default | Description |
|---------|----------|---------|-------------|
| `cluster` | yes | — | Cluster name (e.g., `della`) |
| `model` | yes | — | MLIP family: `mace`, `chgnet`, `uma`, `tensornet` |
| `checkpoint` | no | `default` | Model weights (e.g., `medium`, `uma-s-1p1`) |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `timeout` | no | `120` | Seconds to wait for worker startup |
| `elements` | yes | — | Element symbols mapping atom types (must be last) |

### Notes

- Requires `units metal`. The fix checks this at startup.
- Use `pair_style zero`. The fix provides all interatomic forces.
- Single-node only — the worker sees all atoms and computes its own neighborhoods.
- LAMMPS integration is experimental and has not been tested as thoroughly as the ASE integration yet. If you try it and run into issues, please reach out.

## Setting Up a New Cluster

This section is for people setting up Rootstock on a new cluster. All commands below are run **on the cluster itself** (SSH in first). You'll need write access to a shared filesystem location visible to your users.

### 1. Install Rootstock

On a login node:

```bash
pip install rootstock
```

### 2. Initialize the Rootstock directory

Choose a location on an appropriate shared filesystem where users can read but only maintainers can write. Then run:

```bash
rootstock init
```

This will interactively prompt you for:

- **root** — the shared directory path (e.g., `/scratch/shared/rootstock`)
- **api_key / api_secret** — optional credentials for pushing the cluster manifest to the Rootstock dashboard. Contact a Rootstock maintainer if you want your cluster to appear on the [dashboard](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/). These are [Modal Proxy Auth Tokens](https://modal.com/docs/guide/webhook-proxy-auth).
- **maintainer name / email** — identifies the maintainer for this installation

### 3. Install environments

Still on the login node:

```bash
# Install individual environments
rootstock install mace_env.py --models small,medium
rootstock install chgnet_env.py
rootstock install uma_env.py --models uma-s-1p1
rootstock install tensornet_env.py

# Or point it at a directory with multiple environments
rootstock install ./environments/

# Verify everything is set up
rootstock status
```

Each `rootstock install` command creates an isolated virtual environment under `{root}/envs/`, installs the MLIP's dependencies, and optionally pre-downloads model weights (via `--models`). This can take several minutes per environment depending on the MLIP.

See the [dashboard](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/) for environment files that are known to work — you can use these as a starting point for your cluster.

### 4. Register with the dashboard (optional)

If you configured API credentials during `rootstock init`, the manifest is pushed automatically when you install or update environments. If the push failed (e.g., due to network issues), you can retry:

```bash
rootstock manifest push
```

### Directory Structure

After setup, the rootstock root directory will look like this:

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

The `home/` directory exists because some ML libraries (FAIRChem, MatGL) ignore `XDG_CACHE_HOME` and write to `~/.cache/` unconditionally. Rootstock redirects `HOME` during builds and at worker runtime so that model weights end up in the shared directory rather than in individual users' home directories.

## Writing Environment Files

Each MLIP is defined by a small Python file with [PEP 723](https://peps.python.org/pep-0723/) inline metadata specifying its dependencies and a `setup()` function that returns an ASE calculator:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.14", "ase>=3.22", "torch>=2.0,<2.10"]
# ///

def setup(model: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, device=device, default_dtype="float32")
```

Rootstock uses `uv` to build an isolated virtual environment from these dependencies. The `setup()` function is called once when a worker starts, and the returned calculator is reused for all subsequent calculations in that session.

## Local Development

```bash
git clone https://github.com/Garden-AI/rootstock.git
cd rootstock
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check rootstock/
ruff format rootstock/
```

## Get Involved

Rootstock is an early-stage project and we welcome feedback, bug reports, and collaborators. If you're interested in deploying Rootstock on your cluster, contributing environment files for new MLIPs, or using it for a research project, please contact Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).
