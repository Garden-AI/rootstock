# Run — from other tools

`RootstockCalculator` is a standard ASE calculator. Any tool that accepts an ASE calculator can use a Rootstock-hosted model in its place, without that tool needing the model's Python dependencies. You point the tool at a `RootstockCalculator` for a given `(cluster, checkpoint)` pair, and it talks to the isolated model environment through Rootstock.

The integrations below are planned. This page is a stub and will grow as each one is built out and documented.

## MLIPx

[MLIPx](https://github.com/basf/mlipx) provides recipes for benchmarking and comparing machine-learned interatomic potentials. Because it operates on ASE calculators, Rootstock can supply the calculator for a given checkpoint, letting you run MLIPx comparisons against models installed on a cluster.

Status: planned / not yet documented.

## quacc

[quacc](https://quacc.readthedocs.io) is a workflow engine for high-throughput computational materials science. Rootstock can serve as the calculator backend for quacc recipes, so workflows call a cluster-hosted model instead of bundling its dependencies.

Status: planned / not yet documented.

## Other tools

Want another integration? [Open an issue](https://github.com/Garden-AI/rootstock/issues/new) describing the tool and how you'd use it.
