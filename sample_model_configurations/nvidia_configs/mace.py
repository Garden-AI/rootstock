# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # 0.3.15+ needed for the mh-1 registry entry (matpes needs 0.3.13,
#     # omol needs 0.3.14, mpa-0 needs 0.3.10).
#     "mace-torch>=0.3.15",
#     "ase>=3.22",
#     # 2.4.1 is explicitly unsupported by mace-torch.
#     "torch>=2.4.0,!=2.4.1,<2.10",
# ]
# ///
"""MACE env — hosts MACE-MP-0, MACE-OFF23, MPA-0, MATPES, MH-1, and OMOL checkpoints.

All ship in the same `mace-torch` package, so they share an environment.
Upstream-string routing in CHECKPOINTS: an `off:` prefix routes to mace_off()
and an `omol:` prefix to mace_omol() (float64, molecules only); a `@head`
suffix selects a head of a multi-head model (loaded in float64, per the
MACE-MH-1 model card).

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
    # file mace-mpa-0-medium.model — keep the size explicit like mace-mp-0.
    "mace-mpa-0-medium": "medium-mpa-0",
    "mace-matpes-r2scan-0": "mace-matpes-r2scan-0",
    "mace-mh-1-matpes-r2scan": "mh-1@matpes_r2scan",
    # Only the extra-large OMOL model has been released.
    "mace-omol-0-extra-large": "omol:extra_large",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "mace:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    arg = CHECKPOINTS[checkpoint]
    if arg.startswith("off:"):
        from mace.calculators import mace_off

        return mace_off(model=arg[4:], device=device, default_dtype="float32")
    if arg.startswith("omol:"):
        from mace.calculators import mace_omol

        return mace_omol(model=arg[5:], device=device, default_dtype="float64")
    from mace.calculators import mace_mp

    if "@" in arg:
        model, head = arg.split("@", 1)
        return mace_mp(model=model, device=device, default_dtype="float64", head=head)
    return mace_mp(model=arg, device=device, default_dtype="float32")


def setup_from_path(path: str, device: str = "cuda"):
    # Custom checkpoints (`:custom` ids with user weights): fine-tunes load through
    # MACECalculator directly — the mp/off dispatch in setup() only exists
    # to pick which pretrained file to download.
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=path, device=device, default_dtype="float32")
