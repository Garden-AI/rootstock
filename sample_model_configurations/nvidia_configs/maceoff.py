# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
# ]
# ///
"""MACE-OFF env — hosts MACE-OFF23 checkpoints (organic chemistry)."""

CHECKPOINTS = {
    "mace-off23-small":  "small",
    "mace-off23-medium": "medium",
    "mace-off23-large":  "large",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mace.calculators import mace_off

    return mace_off(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
