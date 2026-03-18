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
