"""
Modal workshop for crafting MLIP configurations on NVIDIA GPUs.

This app is a development sandbox: each MLIP architecture gets its own
explicit `modal.Image` (deps installed via uv) and a probe function that
imports the config's `setup()`, runs one forward pass on a GPU, and prints
structured `STAGE: ...` markers so we can iterate fast and diagnose hangs
without guessing.

It is *not* a validator for finished configs and does not run rootstock
itself. The artifact we ship is the `nvidia_configs/<name>_env.py` file;
this app is the workshop where we tinker until that file is right.

Usage:
    modal run modal_app.py::probe_esen
    modal run modal_app.py::probe_esen --checkpoint esen-sm-conserving-all-oc25 --system slab_co

To add a new MLIP: define a new `<name>_image` and a new `probe_<name>`
function below, following the eSEN block as a template.
"""

from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# Shared infrastructure
# -----------------------------------------------------------------------------

app = modal.App("rootstock-mlip-workshop")

# One volume for *all* model caches. Few large files (weights), so cold-read
# tax is negligible — this is the right place for persistence.
model_cache = modal.Volume.from_name(
    "rootstock-model-cache", create_if_missing=True
)
CACHE_MOUNT = "/cache"

# Cache redirection mirrors what rootstock does on HPC (see top-level README's
# "Directory Structure" section for why HOME redirect is needed for FAIRChem,
# MatGL, and other libraries that ignore XDG_CACHE_HOME). HF_XET_CACHE is
# included because newer huggingface_hub uses xet for delta downloads.
CACHE_ENV = {
    "HOME": f"{CACHE_MOUNT}/home",
    "XDG_CACHE_HOME": CACHE_MOUNT,
    "HF_HOME": f"{CACHE_MOUNT}/huggingface",
    "HF_HUB_CACHE": f"{CACHE_MOUNT}/huggingface/hub",
    "HF_XET_CACHE": f"{CACHE_MOUNT}/huggingface/xet",
}

HERE = Path(__file__).parent
CONFIGS = HERE / "nvidia_configs"
AGENT = HERE / "_agent"

# In-image paths for the config and probe (each image gets its own copy).
IMG_CONFIG = "/workshop/config.py"
IMG_PROBE = "/workshop/probe.py"

HF_SECRET = modal.Secret.from_name("huggingface-token")


def _run_probe_subprocess(checkpoint: str, system: str, device: str = "cuda") -> int:
    """
    Run probe.py as a subprocess so its STAGE markers stream to stdout in
    realtime. Called from inside an @app.function whose image has the right
    deps installed.

    Cache redirection (HOME, HF_HOME, etc.) is applied here at runtime, not
    baked into the image. Baking it would cause uv/pip to write its own cache
    to /cache during image build, leaving files there and blocking the
    volume mount at function start.
    """
    import os
    import subprocess
    import sys

    for path in CACHE_ENV.values():
        os.makedirs(path, exist_ok=True)
    sub_env = {**os.environ, **CACHE_ENV}

    cmd = [sys.executable, IMG_PROBE, "--config", IMG_CONFIG, "--system", system, "--device", device]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    print(f"PROBE_CMD: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=sub_env).returncode


# -----------------------------------------------------------------------------
# eSEN — FAIRChem single-task (OMol25, OC25, ODAC25)
# -----------------------------------------------------------------------------

esen_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "torch>=2.4.0",
        "fairchem-core>=2.0.0",
        "ase>=3.22",
        "torch-geometric",
        find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    )
    .add_local_file(str(CONFIGS / "esen_env.py"), IMG_CONFIG)
    .add_local_file(str(AGENT / "probe.py"), IMG_PROBE)
)


@app.function(
    image=esen_image,
    gpu="A10G",
    volumes={CACHE_MOUNT: model_cache},
    secrets=[HF_SECRET],
    timeout=900,
)
def probe_esen(
    checkpoint: str = "esen-md-direct-all-omol",
    system: str = "molecule",
):
    """Probe an eSEN checkpoint. Default: OMol25 H2O."""
    return _run_probe_subprocess(checkpoint, system)
