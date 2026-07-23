# sample_model_configurations

Curated, working `<mlip>.py` configurations for rootstock — organized by hardware target. Drop one onto an HPC cluster, `rootstock install` it, and run.

## Layout

```
sample_model_configurations/
├── modal_app.py        # the NVIDIA workshop (Modal-based)
├── _agent/             # workshop tooling (probe, failure taxonomy)
│   ├── probe.py
│   ├── failure_modes.md
│   └── README.md
└── nvidia_configs/     # configs targeting NVIDIA GPUs
    ├── allegro.py      # Allegro (NequIP family, custom model required)
    ├── allscaip.py     # AllScAIP (FAIRChem scalable-attention, OMol25)
    ├── ani.py          # ANI-2x / ANI-1ccx / ANI-1x (organic molecules)
    ├── dimenet.py      # DimeNet++ (OC20 catalysis)
    ├── equiformer.py   # EquiformerV2 (OC20 catalysis)
    ├── escn.py         # eSCN (OC20 catalysis)
    ├── esen.py         # eSEN — FAIRChem single-task (OMol25/OC25/ODAC25)
    ├── gemnet.py       # GemNet-OC / GemNet-dT (OC20 catalysis)
    ├── mace.py         # MACE-MP-0/Large + MACE-OFF23 (one mace-torch env)
    ├── mace_polar.py   # MACE-POLAR-1 (electrostatic MACE, git-main + graph-longrange)
    ├── mattersim.py    # MatterSim-v1 (Microsoft universal)
    ├── nequip.py       # NequIP (system-specific, deployed model required)
    ├── orb.py          # Orb v2/v3 (Orbital Materials universal)
    ├── painn.py        # PaiNN (OC20 catalysis)
    ├── schnet.py       # SchNet (OC20 catalysis)
    ├── scn.py          # SCN (OC20 catalysis)
    ├── tensornet.py    # TensorNet (MatPES PBE/r2SCAN via matgl 2.x+HF)
    ├── uma.py          # UMA — FAIRChem multi-task (OMAT/OC/ODAC)
    └── chgnet.py       # CHGNet (inorganic, charge-informed)
```

Other hardware targets (AMD, Apple Silicon, CPU-only) would each get their
own `*_configs/` subfolder when needed. The `_agent/` tooling is
hardware-agnostic.

> **Python floor caveat.** rootstock itself requires Python >=3.11, and the
> worker inside a built env runs rootstock — so every config's
> `requires-python` must admit 3.11. The fairchem-core 1.x configs (dimenet,
> equiformer, escn, gemnet, painn, schnet, scn) pin a single minor
> (`>=3.11,<3.12`) because their torch-scatter/sparse/cluster wheels come
> prebuilt per Python minor from the pinned PyG index; bump the pin in
> lockstep if rootstock's floor moves again.

## What `modal_app.py` is for

It is a **workshop**, not a validator. The artifact we ship is the config
file in `nvidia_configs/`. The workshop is where an agent (or human) tinkers
until they have a config that works on an NVIDIA GPU — the right
dependencies, the right `setup()` call, the right checkpoint name — and then
commits the file.

Each MLIP architecture gets its own explicit `modal.Image` and `probe_*`
function in `modal_app.py`. The image's deps come from `uv pip install`; the
config file is mounted into the image; the probe imports `setup()` from the
config and runs one forward pass on a GPU. Stage markers stream back as
`STAGE: <name> elapsed=<sec>` so hangs are diagnosable.

This deliberately does **not** use `rootstock` itself on Modal. Rootstock is
the HPC deployment story; the workshop is here to get the config file right
fast.

## Iteration loop for a new MLIP

1. Research the MLIP. Find install instructions and the calculator API.
2. Write a candidate `nvidia_configs/<mlip>.py` — PEP 723 deps + a
   `setup(checkpoint, device)` function that returns an ASE calculator. Crib from
   the closest existing config.
3. Add a `probe_<mlip>` function to `modal_app.py` using the `@probe_image()`
   decorator factory (copy any existing probe block; swap config path and
   deps). See **modal_app.py structure** below.
4. `modal run modal_app.py::probe_<mlip>` — watch the STAGE markers.
5. If it fails: read the error, check `_agent/failure_modes.md` for known
   signatures, fix the config or `modal_app.py` deps, re-run.
6. When it passes: commit the config + the modal_app.py addition. Add any
   surprising new failure to `_agent/failure_modes.md`.
7. To deploy: `scp nvidia_configs/<mlip>.py` to your HPC cluster and
   `rootstock install` it. The HPC build path uses the same PEP 723 metadata
   via uv.

## modal_app.py structure

The app uses a `probe_image()` decorator factory that eliminates boilerplate.
Each probe is ~5 lines:

```python
@probe_image(
    "mlip.py",                          # config file in nvidia_configs/
    ["torch>=2.0", "ase>=3.22", "mlip-pkg"],# uv pip install deps
    python_version="3.11",                  # optional (default: "3.11")
    find_links="https://data.pyg.org/...",  # optional PyG wheel index
    apt_packages=["git"],                   # optional apt packages
    no_deps=["pkg @ git+https://..."],      # optional --no-deps installs
    gpu="A10G",                             # optional (default: "A10G")
)
def probe_mlip(checkpoint: str = "default-ckpt", system: str = "crystal"):
    """One-line docstring."""
    return _run_probe_subprocess(checkpoint, system)
```

The `no_deps` parameter is needed when a package has an unresolvable dep
(`lightning` is the known case — see `failure_modes.md`).

**Important**: Modal builds ALL images defined in the app file whenever any
probe is run, not just the target probe's image. A build failure in any one
probe will abort the whole run. Keep every probe's image buildable before
committing.

## Probe systems

| `--system` | Description | Best for |
|------------|-------------|----------|
| `molecule` | H₂O in vacuum | Organic MLIPs (ANI, MACE-OFF23) |
| `crystal`  | 8-atom Cu FCC bulk | Universal inorganic potentials |
| `slab_co`  | Cu(111) 2×2×3 slab + CO adsorbate | Catalysis models (OCP) |

## Required Modal setup

- A `huggingface` Modal secret with read access to any gated model
  repos you'll touch (e.g., FAIRChem checkpoints).
- That's it — the cache volume is created on first run.

## Why a workshop, not a validator?

The mission is *coverage*: a working config for as many MLIPs as we can. The
hard part isn't validating finished configs; it's converging on one. The
workshop optimizes for that — fast image rebuilds via Modal layer caching,
streamed stage markers, a structured taxonomy of failure modes that grows
every time we port a new MLIP. Once we've ported enough, regression-testing
committed configs is a straightforward extension.
