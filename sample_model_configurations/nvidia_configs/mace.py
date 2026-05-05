# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
# ]
# ///
"""MACE env — hosts MACE-MP-0 checkpoints."""

CHECKPOINTS = {
    "mace-mp-0-small":  "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large":  "large",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mace.calculators import mace_mp

    return mace_mp(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
