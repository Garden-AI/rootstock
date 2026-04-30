# `_agent/` — workshop tooling

Scaffolding around the *creation* of `<mlip>_env.py` configs. Hardware-agnostic
— sits at the workshop root because the same probe/taxonomy applies whether
the target is NVIDIA, AMD, or CPU-only.

## Contents

- `probe.py` — runs `setup()` from a config file and does one forward pass on
  a small system, with structured `STAGE: <name> elapsed=<sec>` markers.
  Designed to run inside a Modal image whose deps were installed from the
  config's PEP 723. Takes `--config <path>` to point at the config file
  (mounted into the image at a known path).
- `failure_modes.md` — running taxonomy of failures we've actually hit
  (signature → cause → fix). Update every time porting a new MLIP teaches us
  something new.

## How `probe.py` is used

`modal_app.py` mounts both the chosen config and `probe.py` into each MLIP's
image at fixed paths (e.g., `/workshop/config.py`, `/workshop/probe.py`),
then the probe function shells out to `python /workshop/probe.py --config
/workshop/config.py ...`. The subprocess's stdout streams back to the local
`modal run` log, so `STAGE` markers show up live.

If a stage doesn't fire within ~30s of the previous, the next stage is hung.
That's the whole observability story — keep it boring.

## What doesn't live here

- `<mlip>_env.py` configs — those are the public artifact; they live in
  `nvidia_configs/` (or future `*_configs/` subfolders).
- The Modal app itself — `modal_app.py` is a sibling of this folder.
- Rootstock CLI / server / worker code — that's `rootstock/` at the repo
  root and is not on the workshop's critical path.
