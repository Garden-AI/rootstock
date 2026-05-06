# Rootstock

Rootstock makes it easy to use machine-learned interatomic potentials (MLIPs) on HPC clusters. Researchers can use multiple MLIPs (MACE, CHGNet, UMA, TensorNet, and others) with ASE or LAMMPS without managing conflicting Python environments.

Rootstock provides an [ASE](https://wiki.fysik.dtu.dk/ase/)-compatible calculator that runs each MLIP in an isolated, pre-built Python environment. Swapping models is a one-line change, even when MLIPs require different Python or library versions. Rootstock also integrates with [LAMMPS](https://www.lammps.org/) through a `fix` command for any supported MLIP.

## Status

Rootstock is **early-stage software under active development.** It is currently deployed on two HPC clusters:

- **Della** — Princeton Research Computing
- **Sophia** — Argonne Leadership Computing Facility (ALCF)

We are seeking additional clusters and early users to help shape the tool. If you are interested in deploying Rootstock on your cluster or using it for a specific project, contact Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).

## Quick Start

Rootstock is designed for HPC clusters where it has been set up by a system maintainer. The code below runs in your normal Python environment — inside a SLURM job script, an interactive session, or a Jupyter notebook. Rootstock handles MLIP environment isolation automatically.

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

Swap the underlying potential by changing `checkpoint`: e.g. `checkpoint="uma-s-1p1"` or `checkpoint="tensornet-matpes-pbe-2025-2"`.

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install the lightweight `rootstock` package in your environment.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Learn about the `RootstockCalculator` API and available models.

    [:octicons-arrow-right-24: API](api.md)

-   :material-earth:{ .lg .middle } **Example Configs**

    ---

    See example environment configurations from deployed clusters.

    [:octicons-arrow-right-24: Example Configs](clusters.md)

-   :material-server:{ .lg .middle } **Cluster Setup**

    ---

    Set up Rootstock on a new HPC cluster.

    [:octicons-arrow-right-24: Cluster Setup](cluster-setup.md)

</div>
