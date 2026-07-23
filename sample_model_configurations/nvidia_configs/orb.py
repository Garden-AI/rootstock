# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "orb-models>=0.4.0",
#     "ase>=3.22",
#     "torch>=2.0",
#     # Not imported here — constrains orb-models' transitive dep. setup()'s
#     # no-lock serve path relies on cached_path returning local files without
#     # locking or writing, verified against exactly this version (#67).
#     "cached_path==1.8.10",
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
"""Orb env — hosts Orbital Materials' Orb universal potentials."""

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


def setup(checkpoint: str, device: str = "cuda"):
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    # orb-models exposes one function per checkpoint, e.g. pretrained.orb_v2().
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

    orbff = load_fn(weights_path=str(weights), device=torch.device(device))
    return ORBCalculator(orbff, device=torch.device(device))


def setup_from_path(path: str, device: str = "cuda", arch: str = "orb-v2"):
    # Local checkpoints (`rootstock add-local`). A weights file doesn't say
    # which orb architecture produced it, so `arch` names the pretrained
    # loader to instantiate — register the right one at add-local time
    # (--kwarg arch=...). Handing the loader a local path also means no
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
            f"orb_models.forcefield.pretrained, e.g. {', '.join(CHECKPOINTS)}"
        ) from None

    orbff = load_fn(weights_path=path, device=torch.device(device))
    return ORBCalculator(orbff, device=torch.device(device))
