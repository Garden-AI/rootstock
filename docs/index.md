# Rootstock

Rootstock lets you run many machine-learned interatomic potentials (MLIPs) on an HPC cluster from a single [ASE](https://wiki.fysik.dtu.dk/ase/)-compatible calculator. Each MLIP family runs in its own pre-built, isolated Python environment that a maintainer has already installed and verified on the cluster, so you never resolve conflicting Python or library versions yourself. Swapping models is a one-line change to the `checkpoint` argument.

## How it works

Three figures give the mental model: how you use Rootstock, how it runs a model, and how models get added.

### How do I use Rootstock?

Browse the [Matter Model Almanac](https://garden-ai.github.io/almanac) to find a model that is installed and verified on your cluster, then point a `RootstockCalculator` at it. Each model is already installed in an isolated environment, so changing models does not change your own environment.

![Rootstock user journey: browse the almanac, then run a checkpoint on a cluster](assets/rootstock_user_journey.png)

### How does Rootstock run a model?

You call the calculator from the lightweight `rootstock` library, which carries no model dependencies. Rootstock starts the model in a managed subprocess on the same GPU node, in the environment built for that cluster, loading weights from a cluster-local cache. Positions and forces are exchanged over a local Unix socket using the i-PI protocol. The model is loaded once and kept warm across calls.

![Rootstock runtime: a lightweight client proxies to an isolated model subprocess over a Unix socket](assets/rootstock_runtime.png)

### How does Rootstock add models?

Maintainers define each model family as a Python file: a PEP 723 dependency list, a `CHECKPOINTS` table of canonical ids, and a `setup()` loader that returns an ASE calculator. They build the isolated environment and verify it on a GPU node, and automated re-verification catches regressions over time.

![Rootstock model installation: define a model file, build the env, verify on a GPU node](assets/rootstock_model_installation.png)

## Quick Start

Rootstock runs on clusters where a maintainer has set it up. Your code runs in a normal Python environment on the cluster — a SLURM job, an interactive session, or a Jupyter notebook. Rootstock handles MLIP environment isolation for you.

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
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
```

Use it as a context manager so the worker subprocess is torn down when you are done. Swap the underlying potential by changing `checkpoint`, e.g. `checkpoint="uma-s-1p1"`.

## Driving Rootstock from an agent

There are two ways to use Rootstock. The main path is classical: you write Python (ASE) or a LAMMPS input script and run it on the cluster. The other path is agentic: a coding agent drives Rootstock for you. Rootstock ships an [agent skill](skill/skill.md) that teaches an agent to discover what is deployed and to call `RootstockCalculator` correctly.

## Availability

Rootstock is deployed on a growing set of HPC clusters. Current model-by-cluster coverage — which checkpoints are installed and verified where — lives in the [Matter Model Almanac](https://garden-ai.github.io/almanac) and the live dashboard; see [Clusters](clusters.md).

To deploy Rootstock on your cluster or use it for a specific project, contact Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install the lightweight `rootstock` package in your environment.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Learn the `RootstockCalculator` API and the CLI.

    [:octicons-arrow-right-24: API](api.md)

-   :material-earth:{ .lg .middle } **Clusters**

    ---

    See what is deployed on each cluster and copy a working config.

    [:octicons-arrow-right-24: Clusters](clusters.md)

-   :material-puzzle:{ .lg .middle } **Integrations**

    ---

    Use Rootstock from ASE, LAMMPS, and an agent skill.

    [:octicons-arrow-right-24: Integrations](integrations.md)

-   :material-server:{ .lg .middle } **Cluster Setup**

    ---

    Set up Rootstock on a new HPC cluster.

    [:octicons-arrow-right-24: Cluster Setup](cluster-setup.md)

</div>
