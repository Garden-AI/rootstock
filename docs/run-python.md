# Run — with Python and ASE

The main way to use Rootstock is from Python. You write an [ASE](https://wiki.fysik.dtu.dk/ase/) script and run it on a cluster where a maintainer has set up Rootstock. Your code runs in a normal Python environment on the cluster (a SLURM job, an interactive session, a Jupyter notebook). Rootstock handles MLIP environment isolation for you.

## Install

You install only the lightweight `rootstock` package. It has the `RootstockCalculator` class that knows how to call the models pre-installed on the cluster. You don't need to install heavy model dependencies yourself. Those dependencies (PyTorch, model-specific libraries, and so on) live in the pre-built environments on the cluster.

**Requirements**

- Python 3.11 or later
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
rootstock resolve --cluster delta --json
```

## Quick start

Browse the [Matter Model Almanac](https://garden-ai.github.io/almanac) to find a checkpoint that is installed and verified on your cluster, then point a `RootstockCalculator` at it:

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

Use it as a context manager so the worker subprocess is torn down when you are done. Swap the underlying potential by changing `checkpoint`, e.g. `checkpoint="uma-s-1p1"`.

## Worker lifetime

The `with` block defines the lifetime of the worker subprocess. Entering it spawns the worker and loads the model. The model then stays warm for the duration of the block, so every call inside reuses the loaded weights. Exiting the block tears the worker down and frees the GPU. Scope the block to span all the work you want against that model:

```python
with RootstockCalculator(cluster="delta", checkpoint="mace-mp-0-medium") as calc:
    for atoms in structures:
        atoms.calc = calc
        results.append((atoms.get_potential_energy(), atoms.get_forces()))
```

## Forwarding setup kwargs

Some envs accept extra keyword arguments beyond `checkpoint` and `device`. For example, UMA takes a `task`. Forward them with `setup_kwargs`:

```python
with RootstockCalculator(
    cluster="delta",
    checkpoint="uma-s-1p1",
    setup_kwargs={"task": "omol"},
) as calc:
    ...
```

`setup_kwargs` will not override the top-level `checkpoint` or `device` args. The authoritative list of what an env accepts is its `setup()` signature. You can see what extra arguments a model can take by looking it up in the [Matter Model Almanac](https://garden-ai.github.io/almanac).

## Next steps

- Full constructor parameters, examples, and the CLI: [API Reference](api.md)
- How you get away with not installing the model dependencies yourself: [How does a RootstockCalculator work?](architecture.md)
- Driving Rootstock from other tools or a coding agent: [Run — from other tools](run-tools.md), [Run — with agents](run-agents.md)
