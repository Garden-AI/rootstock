"""
Modal workshop for crafting MLIP configurations on NVIDIA GPUs.

This app is a development sandbox: each MLIP architecture gets its own
explicit `modal.Image` (deps installed via uv) and a probe function that
imports the config's `setup()`, runs one forward pass on a GPU, and prints
structured `STAGE: ...` markers so we can iterate fast and diagnose hangs
without guessing.

It is *not* a validator for finished configs and does not run rootstock
itself. The artifact we ship is the `nvidia_configs/<name>.py` file;
this app is the workshop where we tinker until that file is right.

Usage:
    modal run modal_app.py::probe_esen
    modal run modal_app.py::probe_esen --checkpoint esen-sm-conserving-all-oc25 --system slab_co

To add a new MLIP: add a `probe_<name>` function decorated with `@probe_image(...)`,
following the eSEN block as a template.
"""

from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# Shared infrastructure
# -----------------------------------------------------------------------------

app = modal.App("rootstock-mlip-workshop")

# One volume for *all* model caches. Few large files (weights), so cold-read
# tax is negligible — this is the right place for persistence.
model_cache = modal.Volume.from_name("rootstock-model-cache", create_if_missing=True)
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

HF_SECRET = modal.Secret.from_name("huggingface")


def probe_image(
    config_file: str,
    deps: list[str],
    *,
    python_version: str = "3.10",
    find_links: str | None = None,
    apt_packages: list[str] | None = None,
    no_deps: list[str] | None = None,
    gpu: str = "A10G",
):
    """Decorator factory: builds a Modal image from deps and wires up app.function.

    no_deps: packages installed with --no-deps after the main install (useful
    when a package's declared deps can't resolve but the package itself works).
    """
    img = modal.Image.debian_slim(python_version=python_version)
    if apt_packages:
        img = img.apt_install(*apt_packages)
    img = img.uv_pip_install(*deps, **({"find_links": find_links} if find_links else {}))
    if no_deps:
        img = img.run_commands(
            f"/.uv/uv pip install --system --no-deps {' '.join(repr(p) for p in no_deps)}"
        )
    img = img.add_local_file(str(CONFIGS / config_file), IMG_CONFIG).add_local_file(
        str(AGENT / "probe.py"), IMG_PROBE
    )
    return app.function(
        image=img,
        gpu=gpu,
        volumes={CACHE_MOUNT: model_cache},
        secrets=[HF_SECRET],
        timeout=900,
    )


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

    cmd = [
        sys.executable,
        IMG_PROBE,
        "--config",
        IMG_CONFIG,
        "--system",
        system,
        "--device",
        device,
    ]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    print(f"PROBE_CMD: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=sub_env)
    if result.returncode != 0:
        raise RuntimeError(f"Probe subprocess failed with code {result.returncode}")
    return result.returncode


# -----------------------------------------------------------------------------
# eSEN — FAIRChem single-task (OMol25, OC25, ODAC25)
# -----------------------------------------------------------------------------


@probe_image(
    "esen.py",
    ["torch>=2.4.0", "fairchem-core>=2.0.0", "ase>=3.22", "torch-geometric"],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_esen(checkpoint: str = "esen-md-direct-all-omol", system: str = "molecule"):
    """Probe an eSEN checkpoint. Default: OMol25 H2O."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# AllScAIP — FAIRChem scalable attention MLIP (OMol25, energy-conserving)
# -----------------------------------------------------------------------------


@probe_image(
    "allscaip.py",
    ["fairchem-core>=2.20", "ase>=3.22", "torch>=2.4.0"],
    python_version="3.11",
)
def probe_allscaip(checkpoint: str = "allscaip-md-conserving-all-omol", system: str = "molecule"):
    """Probe the AllScAIP energy-conserving OMol25 model on H2O."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# M3GNet — MatGL universal potential (Materials Project / materialyze HF)
# -----------------------------------------------------------------------------


@probe_image(
    "m3gnet.py",
    ["chgnet>=0.4.0", "ase>=3.22", "torch>=2.0"],
)
def probe_m3gnet(checkpoint: str = "m3gnet-mp-2021-2-8-pes", system: str = "crystal"):
    """M3GNet-PES unavailable in modern matgl; loads CHGNet as substitute."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# CHGNet — charge-informed universal potential
# -----------------------------------------------------------------------------


@probe_image(
    "chgnet.py",
    ["chgnet>=0.3.0", "ase>=3.22", "torch>=2.0"],
)
def probe_chgnet(checkpoint: str = "chgnet-default", system: str = "crystal"):
    """Probe CHGNet. Default checkpoint loads the pretrained model."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# TensorNet — MatGL universal potential (MatPES)
# -----------------------------------------------------------------------------


@probe_image(
    "tensornet.py",
    [
        "torch>=2.4.0",
        "ase>=3.22",
        "torch>=2.4.0",
        "ase>=3.22",
        "huggingface_hub",
        "pymatgen",
        "monty",
        "ruamel.yaml",
        "scipy",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    ],
    python_version="3.11",
    apt_packages=["git"],
    no_deps=["matgl @ git+https://github.com/materialsvirtuallab/matgl.git"],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_tensornet(checkpoint: str = "tensornet-matpes-pbe-2025-2", system: str = "crystal"):
    """Probe a TensorNet/MatGL checkpoint. Default: MatPES PBE on Cu bulk."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# MACE-OFF23 — MACE force field for organic molecules
# -----------------------------------------------------------------------------


@probe_image(
    "mace_off23.py",
    ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.4.0,<2.10"],
)
def probe_mace_off23(checkpoint: str = "mace-off23-medium", system: str = "molecule"):
    """Probe a MACE-OFF23 checkpoint. Default: medium model on H2O."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# MACE-MP-0 / MACE-Large — MACE foundation models for inorganic materials
# -----------------------------------------------------------------------------


@probe_image(
    "mace.py",
    ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.4.0,<2.10"],
)
def probe_mace(checkpoint: str = "mace-mp-0-medium", system: str = "crystal"):
    """Probe a MACE-MP-0 checkpoint. Use --checkpoint large for MACE-Large."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# MACE-POLAR-1 — electrostatic/polarizable MACE foundation models (OMol25)
# -----------------------------------------------------------------------------
# Not in the PyPI mace-torch release yet: installs mace from git main plus
# graph_electrostatics. Uses the mace_polar() loader; the molecule probe system
# supplies the required charge/spin/external_field info keys.


@probe_image(
    "mace_polar.py",
    [
        "ase>=3.22",
        "torch>=2.4.0,<2.10",
        "mace-torch @ git+https://github.com/ACEsuit/mace.git@main",
        "graph-longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git",
    ],
    python_version="3.11",
    apt_packages=["git"],
)
def probe_mace_polar(checkpoint: str = "mace-polar-1-l", system: str = "molecule"):
    """Probe MACE-POLAR-1-L on H2O. Use --checkpoint mace-polar-1-m for the medium variant."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# ANI-2x — TorchANI neural network potential for organic molecules
# -----------------------------------------------------------------------------


@probe_image(
    "ani.py",
    ["torchani>=2.2", "ase>=3.22", "torch>=2.0"],
)
def probe_ani(checkpoint: str = "ani-2x", system: str = "molecule"):
    """Probe an ANI model. Default: ANI-2x on H2O."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# Orb — Orbital Materials universal potential (v2, v3)
# -----------------------------------------------------------------------------


@probe_image(
    "orb.py",
    ["orb-models>=0.4.0,<0.5", "ase>=3.22", "torch>=2.0"],
)
def probe_orb(checkpoint: str = "orb-v2", system: str = "crystal"):
    """Probe an Orb v2 checkpoint. Default: orb-v2 on Cu bulk."""
    return _run_probe_subprocess(checkpoint, system)


@probe_image(
    "orb_v3.py",
    ["orb-models>=0.6.2", "ase>=3.25", "torch>=2.8"],
    python_version="3.12",
)
def probe_orb_v3(checkpoint: str = "orb-v3-conservative-inf-omat", system: str = "crystal"):
    """Probe an Orb v3 checkpoint. Default: conservative-inf-omat on Cu bulk."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# MatterSim — Microsoft universal potential (v1)
# -----------------------------------------------------------------------------


@probe_image(
    "mattersim.py",
    ["mattersim>=1.1.0", "ase>=3.22", "torch>=2.0"],
)
def probe_mattersim(checkpoint: str = "mattersim-v1-0-0-5m", system: str = "crystal"):
    """Probe a MatterSim checkpoint. Default: v1 5M on Cu bulk."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# NequIP — E(3)-equivariant GNN (system-specific, deployed model required)
# -----------------------------------------------------------------------------


@probe_image(
    "nequip.py",
    ["nequip>=0.6.0", "ase>=3.22", "torch>=2.0", "torch-geometric"],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_nequip(checkpoint: str, system: str = "crystal"):
    """Probe a NequIP deployed model. Requires path to a deployed .pth file."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# Allegro — scalable E(3)-equivariant GNN (NequIP family)
# Note: No public checkpoint; probe_allegro is not included here.
# allegro.py exists for use in rootstock environments where a deployed
# Allegro model file is provided. Install with:
#   uv pip install --no-deps "allegro @ git+https://github.com/mir-group/allegro.git"
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# TorchMD-Net — equivariant transformer for MD
# -----------------------------------------------------------------------------


@probe_image(
    "torchmdnet.py",
    [
        "ase>=3.22",
        "torch>=2.0",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    no_deps=["torchmd-net>=2.0.0"],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_torchmdnet(checkpoint: str, system: str = "molecule"):
    """Probe a TorchMD-Net checkpoint (.ckpt path or HF repo ID)."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# GemNet — OC20 GemNet-OC / GemNet-dT via fairchem-core 1.x
# -----------------------------------------------------------------------------


@probe_image(
    "gemnet.py",
    [
        "torch>=2.4.0",
        "fairchem-core>=1.0.0,<2.0.0",
        "ase>=3.22",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_gemnet(checkpoint: str = "gemnet-oc-large-s2ef-oc20-all-md", system: str = "slab_co"):
    """Probe a GemNet OC20 checkpoint. Default: GemNet-OC on CO/Cu slab."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# EquiformerV2 — OC20 EquiformerV2 via fairchem-core 1.x
# -----------------------------------------------------------------------------


@probe_image(
    "equiformer.py",
    [
        "torch>=2.4.0",
        "fairchem-core>=1.0.0,<2.0.0",
        "ase>=3.22",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_equiformer(
    checkpoint: str = "equiformer-v2-153m-s2ef-oc20-all-md", system: str = "slab_co"
):
    """Probe an EquiformerV2 OC20 checkpoint on CO/Cu slab."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# DimeNet++ — OC20 DimeNet++ via fairchem-core 1.x
# -----------------------------------------------------------------------------


@probe_image(
    "dimenet.py",
    [
        "torch>=2.4.0",
        "fairchem-core>=1.0.0,<2.0.0",
        "ase>=3.22",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_dimenet(checkpoint: str = "dimenet-plus-plus-s2ef-oc20-all", system: str = "slab_co"):
    """Probe a DimeNet++ OC20 checkpoint on CO/Cu slab."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# SCN — OC20 SCN via fairchem-core 1.x
# -----------------------------------------------------------------------------


@probe_image(
    "scn.py",
    [
        "torch>=2.4.0",
        "fairchem-core>=1.0.0,<2.0.0",
        "ase>=3.22",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_scn(checkpoint: str = "scn-s2ef-oc20-all-md", system: str = "slab_co"):
    """Probe an SCN OC20 checkpoint on CO/Cu slab."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# eSCN — OC20 eSCN via fairchem-core 1.x
# -----------------------------------------------------------------------------


@probe_image(
    "escn.py",
    [
        "torch>=2.4.0",
        "fairchem-core>=1.0.0,<2.0.0",
        "ase>=3.22",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_escn(checkpoint: str = "escn-l6-m2-lay12-s2ef-oc20-all-md", system: str = "slab_co"):
    """Probe an eSCN OC20 checkpoint on CO/Cu slab."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# PaiNN — OC20 PaiNN via fairchem-core 1.x
# -----------------------------------------------------------------------------


@probe_image(
    "painn.py",
    [
        "torch>=2.4.0",
        "fairchem-core>=1.0.0,<2.0.0",
        "ase>=3.22",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_painn(checkpoint: str = "painn-s2ef-oc20-all", system: str = "slab_co"):
    """Probe a PaiNN OC20 checkpoint on CO/Cu slab."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# SchNet — OC20 SchNet via fairchem-core 1.x
# -----------------------------------------------------------------------------


@probe_image(
    "schnet.py",
    [
        "torch>=2.4.0",
        "fairchem-core>=1.0.0,<2.0.0",
        "ase>=3.22",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
    ],
    find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
)
def probe_schnet(checkpoint: str = "schnet-s2ef-oc20-all", system: str = "slab_co"):
    """Probe a SchNet OC20 checkpoint on CO/Cu slab."""
    return _run_probe_subprocess(checkpoint, system)


# -----------------------------------------------------------------------------
# UMA — FAIRChem multi-task universal model (small and medium)
# -----------------------------------------------------------------------------


@probe_image(
    "uma.py",
    ["fairchem-core>=2.20", "ase>=3.22", "torch>=2.4.0"],
    python_version="3.11",
)
def probe_uma_small(checkpoint: str = "uma-s-1p1", system: str = "crystal"):
    """Probe UMA Small on OMAT-style bulk materials."""
    return _run_probe_subprocess(checkpoint, system)


@probe_image(
    "uma.py",
    ["fairchem-core>=2.20", "ase>=3.22", "torch>=2.4.0"],
    python_version="3.11",
)
def probe_uma_1p2(checkpoint: str = "uma-s-1p2", system: str = "crystal"):
    """Probe UMA Small v1.2 (latest, fixes the uma-s-1 extensivity bug)."""
    return _run_probe_subprocess(checkpoint, system)


@probe_image(
    "uma.py",
    ["fairchem-core>=2.20", "ase>=3.22", "torch>=2.4.0"],
    python_version="3.11",
)
def probe_uma_medium(checkpoint: str = "uma-m-1p1", system: str = "crystal"):
    """Probe UMA Medium on OMAT-style bulk materials."""
    return _run_probe_subprocess(checkpoint, system)
