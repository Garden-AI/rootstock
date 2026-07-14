# Run — from other tools

`RootstockCalculator` is a standard ASE calculator. Any tool that accepts an ASE calculator can use a Rootstock-hosted model in its place, without that tool needing the model's Python dependencies. You point the tool at a `RootstockCalculator` for a given `(cluster, checkpoint)` pair, and it talks to the isolated model environment through Rootstock.

MLIPx is documented below. The remaining integrations are planned and will grow as each one is built out.

## MLIPx

[MLIPx](https://github.com/basf/mlipx) provides recipes for benchmarking and comparing machine-learned interatomic potentials. It evaluates any model that exposes an ASE calculator and does not bundle model code itself, so a Rootstock-hosted checkpoint plugs in the same way MLIPx's other models do.

Install the extra:

```bash
pip install "rootstock[mlipx]"
```

Add a `RootstockMLIPxModel` to a recipe's `models.py`. It records its parameters with zntrack and reports metadata to MLIPx's comparison tables:

```python
from rootstock.integrations.mlipx import RootstockMLIPxModel

MODELS = {
    "mace": RootstockMLIPxModel(checkpoint="mace-mp-0-medium", cluster="sophia", device="cuda"),
    "uma":  RootstockMLIPxModel(checkpoint="uma-s-1p1",        cluster="sophia", device="cuda"),
}
```

Pass `cluster=` for a registered cluster, or `root="/path/to/rootstock"` for a local install (not both). `device` defaults to `cpu`; set `cuda` on GPU nodes. The target environment must already be built with `rootstock install`.

**Cleanup.** Rootstock keeps a worker subprocess alive per calculator and releases it on `close()`. MLIPx does not call `close()`, so for a single model the worker is reaped when the calculator is garbage-collected. For large multi-model comparisons, close calculators explicitly to avoid accumulating workers.

## quacc

[quacc](https://quacc.readthedocs.io) is a workflow engine for high-throughput computational materials science. Rootstock can serve as the calculator backend for quacc recipes, so workflows call a cluster-hosted model instead of bundling its dependencies.

Status: planned / not yet documented.

## atomate2

[atomate2](https://github.com/materialsproject/atomate2) builds materials science workflows out of `Maker` objects, and its force-field Makers run on any ASE calculator. A Rootstock-hosted checkpoint slots in as the calculator, so the workflow itself never needs the model's dependencies.

Install the extra (Python 3.11+, since atomate2 does not support 3.10):

```bash
pip install "rootstock[atomate2]"
```

Use `RootstockAtomate2RelaxMaker` or `RootstockAtomate2StaticMaker` wherever atomate2 expects a `ForceFieldRelaxMaker` or `ForceFieldStaticMaker`:

```python
from jobflow import run_locally

from rootstock.integrations.atomate2 import RootstockAtomate2RelaxMaker

maker = RootstockAtomate2RelaxMaker(
    checkpoint="mace-mp-0-medium",
    cluster="sophia",
    device="cuda",
)
run_locally(maker.make(structure), create_folders=True)
```

They also drop into the forcefield flows (phonons, elastic, EOS, QHA):

```python
from atomate2.forcefields.flows.phonons import PhononMaker

from rootstock.integrations.atomate2 import (
    RootstockAtomate2RelaxMaker,
    RootstockAtomate2StaticMaker,
)

relax = RootstockAtomate2RelaxMaker(checkpoint="mace-mp-0-medium", cluster="sophia")
static = RootstockAtomate2StaticMaker(checkpoint="mace-mp-0-medium", cluster="sophia")

PhononMaker(
    bulk_relax_maker=relax,
    static_energy_maker=static,
    phonon_displacement_maker=static,
)
```

Pass `cluster=` for a registered cluster, or `root="/path/to/rootstock"` for a local install (not both). `device` defaults to `cpu`; set `cuda` on GPU nodes. The target environment must already be built with `rootstock install`.

**On `force_field_name`.** atomate2's `MLFF` enum names models, not execution backends, and rejects unknown strings. A Rootstock-hosted MACE is still MACE, so these Makers leave `force_field_name` at atomate2's `MLFF.Forcefield` placeholder and report the backend through `calculator_meta` instead.

**Cleanup.** atomate2 caches the calculator on the Maker and never calls `close()`, so the worker would outlive the job. These Makers close it themselves once the job finishes. In batch mode that means one worker per structure; pass `close_worker=False` to keep the worker hot across a batch and call `close()` yourself.

## Other tools

Want another integration? [Open an issue](https://github.com/Garden-AI/rootstock/issues/new) describing the tool and how you'd use it.