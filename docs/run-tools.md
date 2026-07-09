# Run — from other tools

`RootstockCalculator` is a standard ASE calculator. Any tool that accepts an ASE calculator can use a Rootstock-hosted model in its place, without that tool needing the model's Python dependencies. You point the tool at a `RootstockCalculator` for a given `(cluster, checkpoint)` pair, and it talks to the isolated model environment through Rootstock.

MLIPx is documented below. The remaining integrations are planned and will grow as each one is built out.

## MLIPx

[MLIPx](https://github.com/basf/mlipx) provides recipes for benchmarking and comparing machine-learned interatomic potentials. It evaluates any model that exposes an ASE calculator and does not bundle model code itself, so a Rootstock-hosted checkpoint plugs in the same way MLIPx's other models do.

Install the extra:

```bash
pip install "rootstock[mlipx]"
```

**Zero-code option.** MLIPx's `GenericASECalculator` loads any calculator by import path. In a recipe's `models.py`:

```python
import mlipx

MODELS = {
    "mace": mlipx.GenericASECalculator(
        module="rootstock",
        class_name="RootstockCalculator",
        kwargs={"checkpoint": "mace-mp-0-medium", "cluster": "sophia"},
        device="cuda",
    ),
}
```

**Tracked node (recommended).** `RootstockModel` records its parameters with zntrack and reports metadata to MLIPx's comparison tables:

```python
from rootstock.mlipx import RootstockModel

MODELS = {
    "mace": RootstockModel(checkpoint="mace-mp-0-medium", cluster="sophia", device="cuda"),
    "uma":  RootstockModel(checkpoint="uma-s-1p1",        cluster="sophia", device="cuda"),
}
```

Pass `cluster=` for a registered cluster, or `root="/path/to/rootstock"` for a local install (not both). `device` defaults to `cpu`; set `cuda` on GPU nodes. The target environment must already be built with `rootstock install`.

**Cleanup.** Rootstock keeps a worker subprocess alive per calculator and releases it on `close()`. MLIPx does not call `close()`, so for a single model the worker is reaped when the calculator is garbage-collected. For large multi-model comparisons, close calculators explicitly to avoid accumulating workers.

## quacc

[quacc](https://quacc.readthedocs.io) is a workflow engine for high-throughput computational materials science. Rootstock can serve as the calculator backend for quacc recipes, so workflows call a cluster-hosted model instead of bundling its dependencies.

Status: planned / not yet documented.

## Other tools

Want another integration? [Open an issue](https://github.com/Garden-AI/rootstock/issues/new) describing the tool and how you'd use it.