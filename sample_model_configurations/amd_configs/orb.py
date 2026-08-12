# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # >=0.5,<0.6: 0.5.5 is what the verified Delta env resolved — the v2
#     # loaders keep their single-return API through 0.5.x. 0.4.x is broken
#     # for us: it imports pynanoflann, which is git-only and undeclared, so
#     # a fresh build dies at import (Delta, 2026-07-31). 0.6 raises the
#     # Python floor to 3.12 — see nvidia_configs/orb_v3.py (no ROCm v3
#     # config yet).
#     "orb-models>=0.5,<0.6",
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
"""Orb v2 env (ROCm) — kept only for the built-in-D3 dispersion variant.

Mirrors nvidia_configs/orb.py: Orb v3 supersedes the other v2 checkpoints,
so the catalog was trimmed to orb-d3-v2 (2026-07-30). There is no ROCm orb
v3 config yet (v3 needs torch>=2.8 ROCm wheels — untested on Frontier).
"""

import os
import shutil
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "orb-d3-v2": "orb-d3-v2",
    # Your own fine-tuned v2-architecture weights: pair with weights=
    # (loaded via setup_from_path). v3 fine-tunes go to orb-v3:custom.
    "orb-v2:custom": None,
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


def setup(checkpoint: str, device: str = "cuda", **kwargs):
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
    return ORBCalculator(orbff, device=torch.device(device), **kwargs)


def setup_from_path(path: str, device: str = "cuda", arch: str = "orb-v2", **kwargs):
    # Custom checkpoints (`:custom` ids with user weights). A weights file doesn't say
    # which orb architecture produced it, so `arch` names the pretrained
    # loader to instantiate — pass the right one at call time
    # (setup_kwargs={"arch": ...} / --kwarg arch=...). Handing the loader a local path also means no
    # network and no cached_path locking (see setup()).
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    fn_name = arch.replace("-", "_")
    try:
        load_fn = getattr(pretrained, fn_name)
    except AttributeError:
        raise ValueError(
            f"unknown orb architecture {arch!r}; expected a loader name from "
            f"orb_models.forcefield.pretrained, e.g. orb-v2, orb-d3-v2"
        ) from None

    orbff = load_fn(weights_path=path, device=torch.device(device))
    return ORBCalculator(orbff, device=torch.device(device), **kwargs)
