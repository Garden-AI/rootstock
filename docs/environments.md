# Writing Environment Files

Each MLIP environment is defined by a small Python file with three pieces:

1. A [PEP 723](https://peps.python.org/pep-0723/) inline metadata block declaring the venv's dependencies.
2. A module-level `CHECKPOINTS: dict[str, str]` table mapping **canonical checkpoint ids** (the slugs that appear on the Almanac, in `rootstock add <id>`, and in `RootstockCalculator(checkpoint=<id>)`) to whatever string the upstream library expects.
3. A `setup(checkpoint, device, ...)` function that looks the id up in `CHECKPOINTS` and returns an ASE calculator.

## Creating a New Environment

```bash
# Scaffold a template in the current directory
rootstock new-env mace

# Specify custom output path
rootstock new-env mace -o ./environments/mace.py

# Overwrite existing file
rootstock new-env mace --force
```

The generated file has a placeholder `CHECKPOINTS` dict and a `setup()` skeleton. Fill in the dependencies, populate `CHECKPOINTS`, and implement `setup()`.

## Basic Structure

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.14", "ase>=3.22", "torch>=2.0,<2.10"]
# ///
"""MACE env — hosts MACE-MP-0 checkpoints."""

CHECKPOINTS = {
    "mace-mp-0-small":  "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large":  "large",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mace.calculators import mace_mp
    return mace_mp(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
```

## How It Works

1. **PEP 723 metadata.** Rootstock uses `uv` to build an isolated venv from the listed dependencies.
2. **`CHECKPOINTS` table.** This is the env's local dispatch table. The keys are canonical ids agreed with the Almanac (one row in the matrix per id). The values are whatever the upstream library wants — could be a short name, a HuggingFace path, a function name, anything.
3. **`setup(checkpoint, device)`.** Called once when a worker starts. The returned calculator is reused for all calculations in that session.

When a user runs `rootstock add mace-mp-0-medium`, Rootstock walks every installed env's `env_source.py`, AST-parses the `CHECKPOINTS` literal, and finds the env that declares the id. A typo errors immediately ("no installed env declares ..."), instead of failing inside `setup()`.

## Required Elements

### PEP 723 Metadata Block

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.14",
#     "ase>=3.22",
#     "torch>=2.0,<2.10"
# ]
# ///
```

- `requires-python`: Minimum Python version.
- `dependencies`: Pip-installable packages with version constraints.

### `CHECKPOINTS` table

Module-level, both keys and values must be string literals. Rootstock AST-parses this — it is read without executing the module.

```python
CHECKPOINTS = {
    "canonical-id-1": "upstream-string-1",
    "canonical-id-2": "upstream-string-2",
}
```

If the upstream string already happens to be a clean canonical id, the mapping is identity:

```python
CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
}
```

### `setup()` function

Signature: `setup(checkpoint: str, device: str = "cuda", **extra)`.

- `checkpoint`: Canonical id; must be a key of `CHECKPOINTS`.
- `device`: PyTorch device.
- Optional extra kwargs are forwarded from `RootstockCalculator(setup_kwargs=...)` and `rootstock add --kwarg KEY=VAL`.

Return: an ASE-compatible calculator.

## Examples

### MACE (MP-0 and OFF23 in one env)

MACE-MP-0 and MACE-OFF23 ship in the same `mace-torch` package, so they share a single env. The `off:` prefix on the upstream string in `CHECKPOINTS` routes to `mace_off()` instead of `mace_mp()` — a small dispatch in `setup()`. Two distinct page identifiers (`mace-mp-0`, `mace-off23`) on the Almanac, six matrix rows, one venv to maintain.

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.4.0,<2.10"]
# ///
"""MACE env — hosts MACE-MP-0 and MACE-OFF23 checkpoints."""

CHECKPOINTS = {
    "mace-mp-0-small":   "small",
    "mace-mp-0-medium":  "medium",
    "mace-mp-0-large":   "large",
    "mace-off23-small":  "off:small",
    "mace-off23-medium": "off:medium",
    "mace-off23-large":  "off:large",
}


def setup(checkpoint: str, device: str = "cuda"):
    arg = CHECKPOINTS[checkpoint]
    if arg.startswith("off:"):
        from mace.calculators import mace_off
        return mace_off(model=arg[4:], device=device, default_dtype="float32")
    from mace.calculators import mace_mp
    return mace_mp(model=arg, device=device, default_dtype="float32")
```

### UMA (FAIRChem)

`setup()` accepts an extra `task` kwarg. Users pass `setup_kwargs={"task": "omol"}` to `RootstockCalculator`, or `--kwarg task=omol` to `rootstock add`.

```python
CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
}


def setup(checkpoint: str, device: str = "cuda", task: str = "omat"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(CHECKPOINTS[checkpoint], device=device)
    return FAIRChemCalculator(predictor, task_name=task)
```

### TensorNet (MatGL via HuggingFace)

The upstream string is a HuggingFace path; the canonical id is the slug shown on the Almanac.

```python
CHECKPOINTS = {
    "tensornet-matpes-pbe-2025-2": "materialyze/TensorNet-PES-MatPES-PBE-2025.2",
}


def setup(checkpoint: str, device: str = "cuda"):
    import matgl
    from huggingface_hub import snapshot_download
    from matgl.ext.ase import PESCalculator

    local_path = snapshot_download(repo_id=CHECKPOINTS[checkpoint])
    return PESCalculator(potential=matgl.load_model(local_path))
```

## Best Practices

### Pin dependency versions

```python
# Good: pinned
# dependencies = ["mace-torch>=0.3.14,<0.4", "torch>=2.0,<2.10"]

# Avoid: unpinned
# dependencies = ["mace-torch", "torch"]
```

### Match canonical ids to the Almanac

The canonical ids in `CHECKPOINTS` are the join key with the Almanac. If the Almanac lists `mace-mp-0-medium` and you ship a `CHECKPOINTS` key of `mace_mp_0_medium`, no row in the matrix lights up. Match the Almanac entry exactly. The Almanac is the public list of canonical ids; this env file is the local dispatch.

### Per-cluster variants

A maintainer with quirky hardware can ship a variant — e.g. `mace_rocm.py` — that supports a strict subset of the canonical ids the standard env declares. Keys that aren't in this env's `CHECKPOINTS` simply won't resolve to it; `rootstock add` finds the right env for each id.

## Testing Your Environment

```bash
# Build the venv
rootstock install my_env.py

# Download and verify a checkpoint by canonical id
rootstock add my-canonical-id

# Inspect the manifest
rootstock status
```

```python
from ase.build import bulk
from rootstock import RootstockCalculator

atoms = bulk("Cu", "fcc", a=3.6)

with RootstockCalculator(
    root="/path/to/rootstock",
    checkpoint="my-canonical-id",
    device="cuda",
) as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy())
```
