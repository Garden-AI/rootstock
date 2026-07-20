# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "orb-models>=0.4.0",
#     "ase>=3.22",
#     "torch>=2.0",
#     # Not imported here - constrains orb-models' transitive dep (see
#     # nvidia_configs/orb.py and Garden-AI/rootstock#67).
#     "cached_path==1.8.10",
#     # torch's ROCm wheels depend on this; it lives only on the ROCm
#     # index, so it must be a direct dep for [tool.uv.sources] to route it.
#     "pytorch-triton-rocm",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-rocm" }
# pytorch-triton-rocm = { index = "pytorch-rocm" }
#
# [[tool.uv.index]]
# name = "pytorch-rocm"
# url = "https://download.pytorch.org/whl/rocm6.4"
# explicit = true
# ///
"""Orb env (ROCm) - Orbital Materials' Orb universal potentials on AMD GPUs."""

import os
import shutil
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "orb-v2": "orb-v2",
    "orb-d3-v2": "orb-d3-v2",
    "orb-mptraj-only-v2": "orb-mptraj-only-v2",
}


def _default_weights_url(load_fn) -> str:
    import inspect

    default = inspect.signature(load_fn).parameters["weights_path"].default
    if not isinstance(default, str) or not default.startswith(("http://", "https://")):
        raise RuntimeError(
            f"{load_fn.__name__} has no URL default for weights_path "
            f"(got {default!r}); update this env file for the installed orb-models"
        )
    return default


def _local_weights_path(url: str) -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "orb" / os.path.basename(url)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def setup(checkpoint: str, device: str = "cuda"):
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    fn_name = CHECKPOINTS[checkpoint].replace("-", "_")
    load_fn = getattr(pretrained, fn_name)

    url = _default_weights_url(load_fn)
    weights = _local_weights_path(url)
    if not weights.exists():
        _fetch(url, weights)

    orbff = load_fn(weights_path=str(weights), device=torch.device(device))
    return ORBCalculator(orbff, device=torch.device(device))
