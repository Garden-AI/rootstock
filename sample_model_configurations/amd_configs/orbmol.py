# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     # 0.7.0 (2026-05-26) introduces orbmol_v2 and the orbmol-v1-* aliases
#     # (orbmol-v1-conservative == orb-v3-conservative-omol). Upstream still
#     # pins dm-tree==0.1.8 (no cp313 wheel), hence the <3.13 cap.
#     "orb-models>=0.7,<0.8",
#     "ase>=3.25",
#     "torch>=2.8,<3.0",
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
"""OrbMol env (ROCm) — Orbital Materials' molecular potentials (OMol25/OPoly26).

orbmol-v1-conservative is upstream's alias for orb-v3-conservative-omol;
orbmol-v2 (2026-05) adds learnable long-range electrostatics (CoulombModule).

ROCm caveats (orb-models 0.7 hard-depends on nvalchemi-toolkit-ops, NVIDIA
Warp/CUDA kernels — pure-python wheel, so it *installs* fine here):

- Neighbor lists: the 0.7 default edge_method is knn_alchemi (Warp/CUDA);
  torch ROCm reports device.type == "cuda", so the default would try to
  launch CUDA kernels on an MI250X. We pin edge_method="knn_scipy" (CPU
  KDTree, pure scipy/torch after that) — override via setup kwarg if a
  ROCm-capable path appears.
- orbmol-v2 electrostatics: non-periodic systems use a pure-torch direct
  Coulomb sum (fine on ROCm). Periodic systems go through nvalchemiops
  Particle Mesh Ewald — expect failure on Frontier. Molecules only.

Charge/spin: these are OMol-style conditioned models — ORBCalculator raises if
atoms.info lacks "charge"/"spin" (spin = multiplicity). Set them per structure
via atoms.info; absent that, we default to neutral singlet (charge=0, spin=1).
"""

import os
import shutil
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "orbmol-v1-conservative": "orbmol-v1-conservative",
    "orbmol-v2": "orbmol-v2",
    # Your own fine-tuned OrbMol weights: pair with weights= (loaded via
    # setup_from_path); pass arch= to pick the base architecture.
    "orbmol:custom": None,
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


def _make_calculator(orbff, atoms_adapter, device, edge_method=None, **kwargs):
    from ase.calculators.calculator import all_changes
    from orb_models.forcefield.inference.calculator import ORBCalculator

    class OrbMolCalculator(ORBCalculator):
        """ORBCalculator that defaults missing charge/spin to neutral singlet."""

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            if atoms is not None:
                atoms.info.setdefault("charge", 0)
                atoms.info.setdefault("spin", 1)
            super().calculate(atoms, properties, system_changes)

    # None = vendor default: here that's the knn_scipy pin (the library
    # default knn_alchemi launches Warp/CUDA kernels — see module docstring);
    # the CUDA variant resolves None to the library default instead. Named in
    # the setup hooks (not just **kwargs) so the signature can't drift
    # between vendor copies.
    if edge_method is None:
        edge_method = "knn_scipy"
    return OrbMolCalculator(orbff, atoms_adapter, device=device, edge_method=edge_method, **kwargs)


def setup(
    checkpoint: str,
    device: str = "cuda",
    precision: str = "float32-high",
    compile: bool | None = None,
    edge_method: str | None = None,
    **kwargs,
):
    # Extra **kwargs go to ORBCalculator (e.g. max_num_neighbors=,
    # half_supercell=); precision/compile go to the checkpoint loader.
    import torch
    from orb_models.forcefield import pretrained

    fn_name = CHECKPOINTS[checkpoint].replace("-", "_")
    load_fn = getattr(pretrained, fn_name)

    # orb-models resolves its default weights URL through `cached_path`, which
    # write-locks its cache dir even on warm hits — EACCES for anyone who can't
    # write the shared install. Handed a *local* path instead, cached_path
    # returns it without locking. So the weights are pre-fetched into the
    # shared model cache at `rootstock add` time (maintainer, cache writable)
    # and every later serve loads that file.
    url = _default_weights_url(load_fn)
    weights = _local_weights_path(url)
    if not weights.exists():
        _fetch(url, weights)

    orbff, atoms_adapter = load_fn(
        weights_path=str(weights),
        device=torch.device(device),
        precision=precision,
        compile=compile,
    )
    return _make_calculator(orbff, atoms_adapter, torch.device(device), edge_method, **kwargs)


def setup_from_path(
    path: str,
    device: str = "cuda",
    arch: str = "orbmol-v2",
    precision: str = "float32-high",
    compile: bool | None = None,
    edge_method: str | None = None,
    **kwargs,
):
    # Custom checkpoints (`:custom` ids with user weights). A weights file
    # doesn't say which architecture produced it, so `arch` names the
    # pretrained loader to instantiate — pass the right one at call time
    # (setup_kwargs={"arch": ...} / --kwarg arch=...). Handing the loader a
    # local path also means no network and no cached_path locking (see setup()).
    import torch
    from orb_models.forcefield import pretrained

    fn_name = arch.replace("-", "_").replace(":", "_")
    try:
        load_fn = getattr(pretrained, fn_name)
    except AttributeError:
        raise ValueError(
            f"unknown OrbMol architecture {arch!r}; expected a loader name from "
            f"orb_models.forcefield.pretrained, e.g. orbmol-v2, orbmol-v1-conservative"
        ) from None

    orbff, atoms_adapter = load_fn(
        weights_path=path,
        device=torch.device(device),
        precision=precision,
        compile=compile,
    )
    return _make_calculator(orbff, atoms_adapter, torch.device(device), edge_method, **kwargs)
