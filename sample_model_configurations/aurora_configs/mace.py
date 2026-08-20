# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # 0.3.15+ needed for the mh-1 registry entry (matpes needs 0.3.13,
#     # omol needs 0.3.14, mpa-0 needs 0.3.10).
#     "mace-torch>=0.3.15",
#     "ase>=3.22",
#     # Intel XPU (Aurora PVC) torch build. >=2.13: older XPU wheels have much
#     # slower FP64 kernels on PVC.
#     "torch>=2.13",
#     # torch's XPU wheels depend on this; it lives only on the XPU index, so it
#     # must be a direct dep for [tool.uv.sources] to route it.
#     "triton-xpu",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-xpu" }
# triton-xpu = { index = "pytorch-xpu" }
#
# [[tool.uv.index]]
# name = "pytorch-xpu"
# url = "https://download.pytorch.org/whl/xpu"
# explicit = true
# ///
"""MACE env (Intel XPU / Aurora) - MACE checkpoints on Intel Data Center GPU Max.

Identical to nvidia_configs/mace.py except torch resolves from the Intel XPU
wheel index and setup() defaults to device="xpu". MACE's plain e3nn/torch path
runs on XPU as-is; cuEquivariance acceleration is CUDA-only and is not used.
Pin one PVC tile with ZE_AFFINITY_MASK in the job (the worker inherits it).

All checkpoints ship in the same `mace-torch` package, so they share an
environment. Upstream-string routing in CHECKPOINTS: an `off:` prefix routes to
mace_off() and an `omol:` prefix to mace_omol() (float64, molecules only); an
`mh:` prefix marks a multi-head model (float64, per the MACE-MH-1 model card).

Multi-head checkpoints select a head via the `head` kwarg on setup()
(setup_kwargs={"head": ...} / --kwarg head=...), named by upstream's training
corpus - see MH1_HEADS; omat_pbe is the default.

The OMOL checkpoint expects `charge` and `spin` in `atoms.info`.
"""

CHECKPOINTS = {
    "mace-mp-0-small": "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large": "large",
    "mace-off23-small": "off:small",
    "mace-off23-medium": "off:medium",
    "mace-off23-large": "off:large",
    # Only a medium MPA-0 has been released, but upstream names the weights
    # file mace-mpa-0-medium.model - keep the size explicit like mace-mp-0.
    "mace-mpa-0-medium": "medium-mpa-0",
    "mace-matpes-r2scan-0": "mace-matpes-r2scan-0",
    # One entry per weights file: MH-1's heads are selected by setup(head=...).
    "mace-mh-1": "mh:mh-1",
    # Only the extra-large OMOL model has been released.
    "mace-omol-0-extra-large": "omol:extra_large",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "mace:custom": None,
}

# MH-1's heads, named by training corpus. The released weights file is the
# authority (mace_select_head --list_heads; the model card also lists a
# rgd1_b3lyp head, but that shipped only in mh-0 - ACEsuit/mace#1462).
# Validated here because upstream only warns on an unknown head and silently
# falls back to the last one.
MH1_HEADS = (
    "omat_pbe",
    "omol",
    "spice_wB97M",
    "oc20_usemppbe",
    "mp_pbe_refit_add",
    "matpes_r2scan",
)


def setup(checkpoint: str, device: str = "xpu", head: str | None = None, **kwargs):
    arg = CHECKPOINTS[checkpoint]
    if arg.startswith("mh:"):
        head = head or "omat_pbe"
        if head not in MH1_HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {', '.join(MH1_HEADS)}")
        from mace.calculators import mace_mp

        kwargs.setdefault("default_dtype", "float64")
        return mace_mp(model=arg[3:], device=device, head=head, **kwargs)
    if head is not None:
        raise ValueError(f"'head' selects a head of a multi-head model; {checkpoint} has one head")
    if arg.startswith("off:"):
        from mace.calculators import mace_off

        kwargs.setdefault("default_dtype", "float32")
        return mace_off(model=arg[4:], device=device, **kwargs)
    if arg.startswith("omol:"):
        from mace.calculators import mace_omol

        kwargs.setdefault("default_dtype", "float64")
        return mace_omol(model=arg[5:], device=device, **kwargs)
    from mace.calculators import mace_mp

    kwargs.setdefault("default_dtype", "float32")
    return mace_mp(model=arg, device=device, **kwargs)


def setup_from_path(path: str, device: str = "xpu", head: str | None = None, **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): fine-tunes load through
    # MACECalculator directly - the mp/off dispatch in setup() only exists
    # to pick which pretrained file to download. `head` is for fine-tunes that
    # keep multiple heads; single-head weights load without it.
    from mace.calculators import MACECalculator

    kwargs.setdefault("default_dtype", "float32")
    return MACECalculator(model_paths=path, device=device, head=head, **kwargs)
