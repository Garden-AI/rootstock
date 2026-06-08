# Run — with Python and ASE

The main way to use Rootstock is from Python: you write an [ASE](https://wiki.fysik.dtu.dk/ase/) script and run it on a cluster where a maintainer has deployed Rootstock. Your code runs in a normal Python environment on the cluster (a SLURM job, an interactive session, a Jupyter notebook). Rootstock handles MLIP environment isolation for you.

## Install

You install only the lightweight `rootstock` package — it carries the `RootstockCalculator`, the i-PI protocol client/server, and the CLI, and nothing else. The heavy model dependencies (PyTorch, CUDA, MACE, CHGNet, FAIRChem, etc.) are **not** installed on your system; they live in the pre-built environments on the cluster, managed by cluster administrators.

**Requirements**

- Python 3.10 or later
- Access to a cluster where Rootstock is deployed, or a custom install root

```bash
pip install rootstock
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install rootstock
```

To confirm the install:

```python
from rootstock import RootstockCalculator

print("Rootstock installed successfully")
```

You can also resolve a cluster's install root from the command line:

```bash
rootstock resolve --cluster della --json
```

## Quick start

Browse the [Matter Model Almanac](https://garden-ai.github.io/almanac) to find a checkpoint that is installed and verified on your cluster, then point a `RootstockCalculator` at it:

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

Use it as a context manager so the worker subprocess is torn down when you are done. Swap the underlying potential by changing `checkpoint`, e.g. `checkpoint="uma-s-1p1"` — changing models does not change your own environment.

## Worker lifetime

The `with` block defines the lifetime of the worker subprocess. Entering it spawns the worker and loads the model — a one-time cost of roughly 5–30 s. The model then stays warm for the duration of the block, so every call inside reuses the loaded weights. Exiting the block tears the worker down and frees the GPU. Scope the block to span all the work you want against that model:

```python
with RootstockCalculator(cluster="della", checkpoint="mace-mp-0-medium") as calc:
    for atoms in structures:
        atoms.calc = calc
        results.append((atoms.get_potential_energy(), atoms.get_forces()))
```

## Forwarding setup kwargs

Some envs accept extra keyword arguments beyond `checkpoint` and `device` — UMA, for example, takes a `task`. Forward them with `setup_kwargs`:

```python
with RootstockCalculator(
    cluster="della",
    checkpoint="uma-s-1p1",
    setup_kwargs={"task": "omol"},
) as calc:
    ...
```

`setup_kwargs` cannot contain `checkpoint` or `device` — those are passed at the top level. The authoritative list of what an env accepts is its `setup()` signature; look it up for a given checkpoint in the [Matter Model Almanac](https://garden-ai.github.io/almanac).

## Next steps

- Full constructor parameters, examples, and the CLI: [API Reference](api.md)
- How the worker subprocess and i-PI socket fit together: [How does a RootstockCalculator work?](architecture.md)
- Driving Rootstock from other tools or a coding agent: [Run — from other tools](run-tools.md), [Run — with agents](run-agents.md)
