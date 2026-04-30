# sample_model_configurations

Curated, working `<mlip>_env.py` configurations for rootstock — organized by hardware target. Drop one onto an HPC cluster, `rootstock install` it, and run.

## Layout

```
sample_model_configurations/
├── modal_app.py        # the NVIDIA workshop (Modal-based)
├── _agent/             # workshop tooling (probe, failure taxonomy)
│   ├── probe.py
│   ├── failure_modes.md
│   └── README.md
└── nvidia_configs/     # configs targeting NVIDIA GPUs
    ├── chgnet_env.py
    ├── esen_env.py
    ├── mace_env.py
    ├── tensornet_env.py
    └── uma_env.py
```

Other hardware targets (AMD, Apple Silicon, CPU-only) would each get their
own `*_configs/` subfolder when needed. The `_agent/` tooling is
hardware-agnostic.

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
2. Write a candidate `nvidia_configs/<mlip>_env.py` — PEP 723 deps + a
   `setup(model, device)` function that returns an ASE calculator. Crib from
   the closest existing config.
3. Add an `<mlip>_image` and `probe_<mlip>` block to `modal_app.py` (copy the
   eSEN block as a template; swap deps and config path).
4. `modal run modal_app.py::probe_<mlip>` — watch the STAGE markers.
5. If it fails: read the error, check `_agent/failure_modes.md` for known
   signatures, fix the config, re-run.
6. When it passes: commit the config + the modal_app.py addition. Add any
   surprising new failure to `_agent/failure_modes.md`.
7. To deploy: `scp nvidia_configs/<mlip>_env.py` to your HPC cluster and
   `rootstock install` it. The HPC build path uses the same PEP 723 metadata
   via uv.

## Required Modal setup

- A `huggingface-token` Modal secret with read access to any gated model
  repos you'll touch (e.g., FAIRChem checkpoints).
- That's it — the cache volume is created on first run.

## Why a workshop, not a validator?

The mission is *coverage*: a working config for as many MLIPs as we can. The
hard part isn't validating finished configs; it's converging on one. The
workshop optimizes for that — fast image rebuilds via Modal layer caching,
streamed stage markers, a structured taxonomy of failure modes that grows
every time we port a new MLIP. Once we've ported enough, regression-testing
committed configs is a straightforward extension.
