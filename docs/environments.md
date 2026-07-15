# Adding Models

A model family is made available to Rootstock through an **environment file**: a small Python file that pins the model's dependencies in an isolated virtual environment and exposes a `setup()` that returns an ASE calculator. One file covers a whole family — every checkpoint it lists.

These files are written once per family and kept as working **samples** in the Rootstock repo, under [`sample_model_configurations/nvidia_configs/`](https://github.com/Garden-AI/rootstock/tree/main/sample_model_configurations/nvidia_configs). Samples are grouped by hardware target; other targets (AMD/ROCm, Apple Silicon, CPU-only) get their own `*_configs/` folder as they are added.

![Define a model family as a Python file, build its isolated env, then verify and re-verify it on a GPU node](assets/rootstock_model_installation.png)

## The flow

Adding a model to a cluster is usually a copy-and-adapt job, not authoring from scratch:

1. **Start from a sample.** Copy the matching `<mlip>.py` from the repo onto the cluster. (Authoring a brand-new family is the exception — see [Writing a file from scratch](#writing-a-file-from-scratch).)
2. **Build and verify.** Run `rootstock install <mlip>.py`, then `rootstock add <checkpoint-id>` to download a checkpoint and verify it with a forward pass on a GPU node.
3. **Adapt to the cluster.** Clusters differ — driver and CUDA versions, the available Python, filesystem quirks — so expect the first verify to surface something. Adjust the dependency pins or `setup()` until it passes, and keep the working file. This iteration is the normal case, not a rare one.

An environment file has three pieces:

1. A [PEP 723](https://peps.python.org/pep-0723/) inline metadata block declaring the venv's dependencies.
2. A module-level `CHECKPOINTS: dict[str, str]` table mapping **canonical checkpoint ids** to whatever string the upstream library expects. A canonical id is the slug used in `rootstock add <id>` and `RootstockCalculator(checkpoint=<id>)`; the [Matter Model Almanac](https://garden-ai.github.io/almanac) registers the same ids so its matrix can join to them.
3. A `setup(checkpoint, device, ...)` function that looks the id up in `CHECKPOINTS` and returns an ASE calculator.

## Writing a file from scratch

When no sample exists for a model family yet, scaffold a fresh file:

```bash
# Scaffold a template in the current directory
rootstock new-env mace

# Specify custom output path
rootstock new-env mace -o ./environments/mace.py

# Overwrite existing file
rootstock new-env mace --force
```

The generated file has a placeholder `CHECKPOINTS` dict and a `setup()` skeleton. Fill in the dependencies, populate `CHECKPOINTS`, and implement `setup()` — the rest of this page describes what goes in each piece. Once it works, contribute it back as a sample so the next cluster can copy it.

## Basic structure

```python
# /// script
# requires-python = ">=3.11"
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

## How it works

1. **PEP 723 metadata.** Rootstock uses `uv` to build an isolated venv from the listed dependencies.
2. **`CHECKPOINTS` table.** This is the env's local dispatch table. The keys are canonical ids; the Almanac registers the same ids as its join key. The values are whatever the upstream library wants — a short name, a HuggingFace path, a function name, anything.
3. **`setup(checkpoint, device)`.** Called once when a worker starts. The returned calculator is reused for all calculations in that session.

When a user runs `rootstock add mace-mp-0-medium`, Rootstock walks every installed env's `env_source.py`, AST-parses the `CHECKPOINTS` literal, and finds the env that declares the id. A typo errors immediately ("no installed env declares ..."), instead of failing inside `setup()`.

## Lockfiles and reproducible rebuilds

The PEP 723 block declares version *ranges*; the exact package set is resolved once, at build time. `rootstock install` records that resolution in a uv lockfile so a rebuild reproduces the env instead of re-resolving whatever the ranges allow that day:

- `{root}/environments/<name>.py.lock` — the working lockfile, next to the registered source. `uv lock --script` writes it on first build and keeps its pins on later builds.
- `{root}/envs/<name>/env_source.py.lock` — a copy stored inside the built env, recording exactly what that build was resolved from. Its hash is tracked in the manifest as `lock_hash`.

Rebuilds (`rootstock install <name> --force`) install exactly the locked versions by default. This is what makes "roll out a small fix and rebuild" safe: the rebuilt env has the same dependency stack that was already qualified on the cluster. Two things change the resolution:

- **Editing the env source.** Changed constraints re-resolve minimally; pins that still satisfy the ranges are kept.
- **`rootstock install <name> --force --upgrade`.** Re-resolves everything to the latest allowed versions. Use this when you deliberately want a fresh stack — and expect to re-verify checkpoints afterwards.

If you keep env files in a git repo, commit the `.py.lock` next to the `.py`: `rootstock install ./mace.py` carries an adjacent lockfile along and builds from it. Envs built before lockfiles existed (manifest `lock_hash: null`) can only be re-resolved, not faithfully rebuilt.

**Not every env can be locked.** `uv lock` resolves for every platform at once, so an env pulling prebuilt wheels from a platform-specific index — the PyG `find-links` pages used by the fairchem-core 1.x configs ship no macOS wheels — fails universal resolution. `install` warns and builds it without a lockfile (a plain current-platform resolution, exactly the pre-lockfile behavior); such envs stay `lock_hash: null` and re-resolve on every rebuild.

## Required elements

### PEP 723 metadata block

```python
# /// script
# requires-python = ">=3.11"
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

MACE-MP-0 and MACE-OFF23 ship in the same `mace-torch` package, so they share a single env. The `off:` prefix on the upstream string in `CHECKPOINTS` routes to `mace_off()` instead of `mace_mp()` — a small dispatch in `setup()`.

```python
# /// script
# requires-python = ">=3.11"
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

The upstream string is a HuggingFace path; the canonical id is a short slug.

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

## Best practices

### Pin dependency versions

```python
# Good: pinned
# dependencies = ["mace-torch>=0.3.14,<0.4", "torch>=2.0,<2.10"]

# Avoid: unpinned
# dependencies = ["mace-torch", "torch"]
```

Ranges bound what a *fresh* resolution may pick; the build-time lockfile (see [Lockfiles and reproducible rebuilds](#lockfiles-and-reproducible-rebuilds)) is what pins a given install's rebuilds exactly.

### Match canonical ids to the Almanac

The canonical ids in `CHECKPOINTS` are the join key with the Almanac. If the Almanac registers `mace-mp-0-medium` and you ship a `CHECKPOINTS` key of `mace_mp_0_medium`, the two never join and no row in the matrix lights up. Match the registered id exactly. The Almanac is the registry of canonical ids; this env file is the local dispatch.

### Serve time must not write to the shared install

`rootstock add` (run by a maintainer, who can write the shared cache) is when weights download; after that, `setup()` runs as arbitrary users who can only *read* the install. So `setup()` must not write under the shared root on a warm cache — no lock files, no re-downloads, no "touch to check". Libraries that take a write-lock even on cache hits (e.g. `cached_path`, which orb-models uses) break this: hand them a local file path instead of a URL, pre-fetching the file into `$XDG_CACHE_HOME` yourself — see `nvidia_configs/orb.py`. Runtime scratch (compiled kernels, config dirs) is already redirected per-user by rootstock; this rule is about what your `setup()` and its libraries do with model files.

### Expect cluster-specific edits

The same model rarely drops onto every cluster unchanged. Driver and CUDA versions, the available Python, and filesystem behavior all vary, so adapting a sample's dependency pins or `setup()` for a given cluster is routine, not exceptional. A file can also declare a strict subset of the canonical ids the standard sample carries — keys it doesn't list simply won't resolve to it, and `rootstock add` finds the right env for each id.

When an entire hardware class needs a different dependency stack (a non-NVIDIA GPU, say), that belongs in its own sample folder alongside `nvidia_configs/`, rather than as a one-off edit to an existing file.

## Testing your environment

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
