# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "orb-models>=0.6.2",
#     "ase>=3.25",
#     "torch>=2.8",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Orb v3 env — Orbital Materials' Orb v3 universal potentials.

Separate from orb.py because orb-models>=0.5 changed the loader API
(returns a tuple, requires `atoms_adapter` on ORBCalculator, moved calculator
import path) and 0.6.x bumped the Python floor to 3.12 and torch to 2.8.
"""

import os
import shutil
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "orb-v3-conservative-inf-omat": "orb-v3-conservative-inf-omat",
    "orb-v3-conservative-20-omat":  "orb-v3-conservative-20-omat",
    "orb-v3-direct-inf-omat":       "orb-v3-direct-inf-omat",
    "orb-v3-direct-20-omat":        "orb-v3-direct-20-omat",
    "orb-v3-conservative-inf-mpa":  "orb-v3-conservative-inf-mpa",
    "orb-v3-conservative-20-mpa":   "orb-v3-conservative-20-mpa",
    "orb-v3-direct-inf-mpa":        "orb-v3-direct-inf-mpa",
    "orb-v3-direct-20-mpa":         "orb-v3-direct-20-mpa",
    "orb-v3-conservative-omol":     "orb-v3-conservative-omol",
    "orb-v3-direct-omol":           "orb-v3-direct-omol",
}


def _default_weights_url(load_fn) -> str:
    """The upstream URL baked into the loader's ``weights_path`` default."""
    import inspect

    default = inspect.signature(load_fn).parameters["weights_path"].default
    if not isinstance(default, str) or not default.startswith(("http://", "https://")):
        raise RuntimeError(
            f"{load_fn.__name__} has no URL default for weights_path "
            f"(got {default!r}); update this env file for the installed orb-models"
        )
    return default


def _local_weights_path(url: str) -> Path:
    """Where the checkpoint lives in the shared model cache."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "orb" / os.path.basename(url)


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically (tmp file + rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def setup(checkpoint: str, device: str = "cuda", precision: str = "float32-high"):
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    fn_name = CHECKPOINTS[checkpoint].replace("-", "_")
    load_fn = getattr(pretrained, fn_name)

    # orb-models resolves its default weights URL through `cached_path`, which
    # write-locks its cache dir even on warm hits — EACCES for anyone who can't
    # write the shared install (Garden-AI/rootstock#67). Handed a *local* path
    # instead, cached_path returns it without locking. So the weights are
    # pre-fetched into the shared model cache at `rootstock add` time
    # (maintainer, cache writable) and every later serve loads that file.
    url = _default_weights_url(load_fn)
    weights = _local_weights_path(url)
    if not weights.exists():
        _fetch(url, weights)

    orbff, atoms_adapter = load_fn(
        weights_path=str(weights), device=torch.device(device), precision=precision
    )
    return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=torch.device(device))
