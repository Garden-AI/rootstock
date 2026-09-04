# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "deepmd-kit[torch]>=3.2.0",
#     "ase>=3.23",
#     # deepmd-kit's torch extra pins torch==2.11.0.*. Listed here too so the
#     # explicit cu128 index below routes it: an `explicit` index is only
#     # consulted for packages named in `dependencies`, and PyPI's torch 2.11
#     # wheels bundle CUDA 13, which needs a newer driver than most clusters run.
#     "torch>=2.11,<2.12",
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
"""DeePMD-kit env — hosts the DPA foundation models built into deepmd-kit.

The upstream strings are the names deepmd-kit's own pretrained-model registry
knows (the same ones ``dp pretrained download`` accepts). Weights are fetched
from Hugging Face with sha256 verification into the shared model cache and
loaded from that local file, so serving needs no network.

Multitask checkpoints (the OpenLAM DPA-2.4 / DPA-3.x models) carry one
fitting head per training dataset and require ``head=`` selecting one —
``setup_kwargs={"head": "OMat24"}`` on RootstockCalculator, or
``--kwarg head=OMat24`` for ``rootstock add``. Valid heads per checkpoint
are listed in HEADS; pick the dataset closest to your system and check it
covers your elements. Single-task checkpoints take no head.

Charge and spin: checkpoints trained with frame-level charge/spin inputs
(DPA3-Omol-Large, DPA-3.2-5M, DPA-3.3-1M) read them from
``atoms.info["charge"]`` and ``atoms.info["spin"]`` (spin = multiplicity
2S+1), the convention the other OMol-trained envs use. Without them the
model's trained-in default (neutral singlet) applies. deepmd's native
``fparam`` / ``charge_spin`` info keys still work and take precedence.

Device: deepmd's PyTorch backend fixes its device at first import from the
environment (``DEVICE=cpu`` for CPU, otherwise ``cuda:{LOCAL_RANK}``), so
setup() sets those variables before importing deepmd.

MPI: the deepmd-kit wheel preloads ``libmpi.so.12`` from the ``mpich`` wheel
at import (its custom-op library links MPI, used only by the LAMMPS
plugin). That libmpi needs the libfabric bundled next to it under
``lib/mpich/``; on Cray systems the module environment puts the system
libfabric on ``LD_LIBRARY_PATH``, which outranks the wheel's RUNPATH and
lacks the ``FABRIC_1.9`` symbol version libmpi wants. setup() therefore
loads the bundled libfabric by absolute path first, so the dynamic linker
reuses it when libmpi asks for ``libfabric.so.1``.

Licenses: the DPA-2 / DPA-3 checkpoints are CC-BY-4.0; the DPA4 checkpoints
are CC-BY-NC-4.0 (non-commercial).
"""

import ctypes
import os
import sys
from importlib import metadata
from pathlib import Path

CHECKPOINTS = {
    # Single-task checkpoints: no head selection.
    "dpa3-omol-large": "DPA3-Omol-Large",
    "dpa4-nano-omat24": "DPA4-Nano-OMat24-v20260805",
    "dpa4-mini-omat24": "DPA4-Mini-OMat24-v20260805",
    "dpa4-neo-omat24": "DPA4-Neo-OMat24-v20260805",
    "dpa4-air-omat24": "DPA4-Air-OMat24-v20260805",
    "dpa4-plus-omat24": "DPA4-Plus-OMat24-v20260805",
    # Multitask OpenLAM checkpoints: head= required (see HEADS).
    "dpa-3.3-1m": "DPA-3.3-1M",
    "dpa-3.2-5m": "DPA-3.2-5M",
    "dpa-3.1-3m": "DPA-3.1-3M",
    "dpa-2.4-7m": "DPA-2.4-7M",
    # Your own DeePMD-kit model — a training checkpoint (.pt) or frozen model
    # (.pth): pair with weights= (loaded via setup_from_path).
    "dpa:custom": None,
}

# Fitting heads of the multitask checkpoints (the upstream `model-branch`
# names). Aliases such as "Default" or "materials" are accepted too, and
# matching is case-insensitive.
_OPENLAM_V2_HEADS = (
    "OMat24",
    "OMol25",
    "MPTrj",
    "OC20M",
    "OC22",
    "ODAC23",
    "Alex2D",
    "MPGen_OpenCSP",
    "Domains_Alloy",
    "Domains_Anode",
    "Domains_Cluster",
    "Domains_FerroEle",
    "Domains_SSE_PBE",
    "Domains_SemiCond",
    "H2O_H2O_PD",
    "Metals_AlMgCu",
    "Metals_AgAu_PBED3",
    "Others_In2Se3",
    "Alloy_APEX",
    "SSE_ABACUS",
    "Hybrid_Perovskite",
    "Electrolyte",
    "Organic_Reactions",
)
_OPENLAM_V1_HEADS = (
    "Omat24",
    "MP_traj_v024_alldata_mixu",
    "OC20M",
    "OC22",
    "ODAC23",
    "Alex2D",
    "SPICE2",
    "Domains_Alloy",
    "Domains_Anode",
    "Domains_Cluster",
    "Domains_Drug",
    "Domains_FerroEle",
    "Domains_SSE_PBE",
    "Domains_SSE_PBESol",
    "Domains_SemiCond",
    "Domains_Transition1x",
    "H2O_H2O_PD",
    "Metals_AlMgCu",
    "Metals_Sn",
    "Metals_Ti",
    "Metals_V",
    "Metals_W",
    "Metals_AgAu_PBED3",
    "Others_HfO2",
    "Others_In2Se3",
    "Alloy_tongqi",
    "SSE_ABACUS",
    "Hybrid_Perovskite",
    "solvated_protein_fragments",
    "Electrolyte",
    "Organic_Reactions",
)
HEADS = {
    "dpa-3.3-1m": _OPENLAM_V2_HEADS,
    "dpa-3.2-5m": _OPENLAM_V2_HEADS,
    "dpa-3.1-3m": _OPENLAM_V1_HEADS,
    "dpa-2.4-7m": _OPENLAM_V1_HEADS,
}

# Head used when verification runs with no user selection. Users still select
# explicitly. The custom entry's value is forwarded to whichever shipped
# checkpoint the weights= leg borrows: single-task checkpoints ignore it.
VERIFY_KWARGS = {
    "dpa-3.3-1m": {"head": "OMat24"},
    "dpa-3.2-5m": {"head": "OMat24"},
    "dpa-3.1-3m": {"head": "Omat24"},
    "dpa-2.4-7m": {"head": "Omat24"},
    "dpa:custom": {"head": "OMat24"},
}


def _cache_dir() -> Path:
    """Where the registry downloads land: deepmd's default layout, relocated
    into the shared model cache."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache / "deepmd" / "pretrained" / "models"


def _select_device(device: str) -> None:
    """Point deepmd's PyTorch backend at ``device``.

    The backend reads its device once, when ``deepmd.pt.utils.env`` is first
    imported: ``DEVICE=cpu`` forces CPU, otherwise it uses
    ``cuda:{LOCAL_RANK}`` (LOCAL_RANK defaulting to 0). There is no per-call
    device argument, so this must run before the first deepmd import.
    """
    if device == "cpu":
        os.environ["DEVICE"] = "cpu"
    elif device.startswith("cuda:"):
        os.environ["LOCAL_RANK"] = device.split(":", 1)[1]


def _bundled_libfabric() -> Path | None:
    """The libfabric shipped inside the ``mpich`` wheel, if installed."""
    try:
        files = metadata.files("mpich") or []
    except metadata.PackageNotFoundError:
        files = []
    for entry in files:
        if entry.match("mpich/libfabric.so.1"):
            return Path(entry.locate()).resolve()
    fallback = Path(sys.prefix) / "lib" / "mpich" / "libfabric.so.1"
    return fallback if fallback.is_file() else None


def _preload_bundled_libfabric() -> None:
    """Load the wheel's own libfabric before deepmd pulls in libmpi.

    Must run before the first deepmd import: once ``libmpi.so.12`` has
    resolved ``libfabric.so.1`` against whatever LD_LIBRARY_PATH offered
    (the Cray system copy, on Delta), the choice is fixed for the process.
    A library already loaded under that soname is reused instead.
    """
    lib = _bundled_libfabric()
    if lib is not None:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _calculator_class():
    """deepmd's ASE calculator, reading charge/spin the way the other envs do."""
    from ase.calculators.calculator import all_changes
    from deepmd.calculator import DP

    class RootstockDP(DP):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Two ways a DPA model can take charge/spin: as the two frame
            # parameters (DPA3-Omol-Large, DPA-3.2-5M) or through a dedicated
            # charge/spin embedding (DPA-3.3-1M, DPA4-OMol).
            self._charge_spin_as_fparam = self.dp.get_dim_fparam() == 2
            self._charge_spin_embedded = self.dp.has_chg_spin_ebd()

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces", "virial"),
            system_changes=all_changes,
        ):
            if atoms is not None and ("charge" in atoms.info or "spin" in atoms.info):
                atoms = atoms.copy()
                charge_spin = [
                    float(atoms.info.get("charge", 0)),
                    float(atoms.info.get("spin", 1)),
                ]
                if self._charge_spin_as_fparam:
                    atoms.info.setdefault("fparam", charge_spin)
                if self._charge_spin_embedded:
                    atoms.info.setdefault("charge_spin", charge_spin)
            super().calculate(atoms, list(properties), system_changes)

    return RootstockDP


def setup(checkpoint: str, device: str = "cuda", head: str | None = None, **kwargs):
    """
    Load a DeePMD-kit calculator for a built-in pretrained model.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu").
        head: Fitting head of a multitask checkpoint (see HEADS). Required
            for those; ignored by single-task checkpoints.

    Returns:
        ASE-compatible calculator.
    """
    if checkpoint in HEADS and head is None:
        raise ValueError(
            f"{checkpoint} is multitask and has no default head - select one "
            f'with setup_kwargs={{"head": ...}}: one of {", ".join(HEADS[checkpoint])}'
        )
    _select_device(device)
    _preload_bundled_libfabric()

    from deepmd.pretrained.download import resolve_model_path

    # Resolves the registry name to a sha256-verified local file, downloading
    # only when it is missing from the cache (at `rootstock add` time).
    weights = resolve_model_path(CHECKPOINTS[checkpoint], cache_dir=_cache_dir())
    return _calculator_class()(model=str(weights), head=head, **kwargs)


def setup_from_path(path: str, device: str = "cuda", head: str | None = None, **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): DP loads a training
    # checkpoint (.pt) or a frozen model (.pth) straight from a path. A
    # multitask fine-tune needs its head (setup_kwargs={"head": ...} /
    # --kwarg head=...) unless it declares a default; single-task ones
    # take none.
    _select_device(device)
    _preload_bundled_libfabric()
    return _calculator_class()(model=path, head=head, **kwargs)
