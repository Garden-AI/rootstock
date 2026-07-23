# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
# ]
# ///
"""MACE env — hosts MACE-MP-0 and MACE-OFF23 checkpoints.

Both ship in the same `mace-torch` package, so they share an environment.
The `off:` prefix on the upstream string in CHECKPOINTS routes to mace_off()
instead of mace_mp().
"""

CHECKPOINTS = {
    "mace-mp-0-small": "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large": "large",
    "mace-off23-small": "off:small",
    "mace-off23-medium": "off:medium",
    "mace-off23-large": "off:large",
}


def setup(checkpoint: str, device: str = "cuda"):
    arg = CHECKPOINTS[checkpoint]
    if arg.startswith("off:"):
        from mace.calculators import mace_off

        return mace_off(model=arg[4:], device=device, default_dtype="float32")
    from mace.calculators import mace_mp

    return mace_mp(model=arg, device=device, default_dtype="float32")


def setup_from_path(path: str, device: str = "cuda"):
    # Local checkpoints (`rootstock add-local`): fine-tunes load through
    # MACECalculator directly — the mp/off dispatch in setup() only exists
    # to pick which pretrained file to download.
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=path, device=device, default_dtype="float32")
