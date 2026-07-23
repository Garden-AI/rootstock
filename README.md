# Rootstock

Rootstock lets you run many machine-learned interatomic potentials (MLIPs) on an HPC cluster from a single [ASE](https://wiki.fysik.dtu.dk/ase/)-compatible calculator, without managing the conflicting Python environments each MLIP requires.

Each MLIP family runs in its own pre-built, isolated Python environment that a maintainer has already installed and verified on the cluster. Swapping models is a one-line change to the `checkpoint` argument, even when the MLIPs need different Python or library versions.

Full documentation: [garden-ai.github.io/rootstock](https://garden-ai.github.io/rootstock/). See which models are supported on which clusters on the [Matter Model Almanac](https://garden-ai.github.io/almanac).

## Support

This work is supported by NSF Award [#2514142](https://www.nsf.gov/awardsearch/show-award?AWD_ID=2514142): “Collaborative Research: Frameworks: SINAPSE: Scalable Infrastructure for AI-coupled Predictive Simulation Enhancement”

## Availability

Rootstock is deployed on a growing set of HPC clusters. Which checkpoints are installed where lives in the [Matter Model Almanac](https://garden-ai.github.io/almanac). If your cluster isn't supported but you want it to be, please reach out to Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).

## Quick Start

Rootstock is designed for use on an HPC cluster where models have been set up by a maintainer. The code below runs in a Python environment with rootstock and ASE installed. This could be inside a SLURM job script, an interactive session, a Jupyter notebook on the cluster, etc..

```python
from ase.build import bulk
from rootstock import RootstockCalculator

atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

with RootstockCalculator(
    cluster="delta",
    checkpoint="mace-mp-0-medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
```

Swap the underlying potential by changing `checkpoint`, e.g. `checkpoint="uma-s-1p1"`.

## Installation

Users install only the lightweight `rootstock` package. The heavy ML dependencies (PyTorch, MACE, FAIRChem, etc.) live in the pre-built environments on the cluster.

```bash
pip install rootstock
# or
uv pip install rootstock
```

## API

`checkpoint` is the canonical id of a specific set of trained weights (e.g. `mace-mp-0-medium`, `uma-s-1p1`). The hosting environment is resolved automatically — Rootstock walks the installed envs and finds the one whose `CHECKPOINTS` table declares the id.

```python
# Pick a checkpoint by canonical id; env is resolved automatically.
RootstockCalculator(cluster="delta", checkpoint="mace-mp-0-medium")

# Forward extra kwargs to the env's setup() function:
RootstockCalculator(cluster="delta", checkpoint="uma-s-1p1", setup_kwargs={"task": "omol"})

# Custom root path instead of a known cluster
RootstockCalculator(root="/scratch/gpfs/specific/install/path/rootstock", checkpoint="mace-mp-0-medium")
```

## Available Models

What is deployed and verified per cluster changes over time. The [Matter Model Almanac](https://garden-ai.github.io/almanac) and the [Clusters](https://garden-ai.github.io/rootstock/clusters/) page show the current coverage.

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

Rootstock ships an experimental LAMMPS `fix` that spawns a worker subprocess, giving a LAMMPS run access to a Rootstock-managed MLIP for molecular dynamics. It is far less tested than the ASE path. See [LAMMPS Integration](https://garden-ai.github.io/rootstock/lammps/) in the docs for the fix syntax and current limitations.

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

Still on the login node — `install` only builds the venv:

```bash
# Install individual environments
rootstock install mace.py
rootstock install uma.py
rootstock install tensornet.py

# Or point it at a directory with multiple environments
rootstock install ./environments/
```

Each `rootstock install` command creates an isolated virtual environment under `{root}/envs/` and installs the MLIP's dependencies. This can take several minutes per environment depending on the MLIP.

### 4. Add checkpoints

Use `rootstock add` to download model weights and verify them on the GPU. Download and verify can run on different nodes — useful when your GPU node has no network access:

```bash
# Login node (CPU, has network): download only
rootstock add mace-mp-0-medium --no-verify
rootstock add uma-s-1p1 --no-verify

# GPU node: skip download (already fetched), verify on GPU
rootstock add mace-mp-0-medium
rootstock add uma-s-1p1

# Forward extra kwargs to setup() — values are JSON-decoded, fall back to strings
rootstock add uma-s-1p1 --kwarg task=omat
```

`rootstock add` is idempotent. Use `rootstock smoke-test` to re-verify all fetched checkpoints (suitable for nightly cron with `--json`).

`rootstock status` shows a per-checkpoint grid of fetched/verified/stale state. See the [dashboard](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/) for environment files that are known to work.

Users can also bring their own weights (e.g. a fine-tuned UMA model) without any write access to the shared install — `rootstock add-local` registers a local file under a checkpoint id that then works everywhere a canonical id does:

```bash
rootstock add-local /scratch/me/my-uma-ft.pt --env uma --id my-uma-ft --kwarg task=omol
rootstock remove-local my-uma-ft   # delete the registration (never the file)
```

The registration lives in the per-user registry (`~/.config/rootstock/local-checkpoints.json`); the hosting env must declare a `setup_from_path()` function (see [docs/environments.md](docs/environments.md)).

### 5. Register with the dashboard (optional)

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

The `home/` directory exists because some ML libraries (FAIRChem, MatGL) ignore `XDG_CACHE_HOME` and write to `~/.cache/` unconditionally. Rootstock redirects `HOME` during builds and at worker runtime so that model weights end up in the shared directory rather than in individual users' home directories.

## Writing Environment Files

Each MLIP is defined by a small Python file with [PEP 723](https://peps.python.org/pep-0723/) inline metadata for its dependencies, a `CHECKPOINTS` table mapping canonical checkpoint ids to whatever string the upstream library expects, and a `setup()` function that dispatches via that table and returns an ASE calculator:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["mace-torch>=0.3.14", "ase>=3.22", "torch>=2.0,<2.10"]
# ///

CHECKPOINTS = {
    "mace-mp-0-small":  "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large":  "large",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
```

`CHECKPOINTS` is the env's local dispatch table. The keys are the canonical ids that show up in `rootstock add <id>` and in `RootstockCalculator(checkpoint=<id>)`; the values are whatever string the upstream library wants. A typo in the canonical id errors immediately ("no installed env declares ...") instead of failing inside `setup()`.

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

We welcome feedback, bug reports, and collaborators. If you're interested in deploying Rootstock on your cluster, contributing environment files for new MLIPs, or using it for a research project, contact Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).
