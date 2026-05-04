# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
# ]
# ///
"""
MACE-OFF environment for Rootstock.

MACE-OFF is the organic-chemistry-focused MACE foundation model, trained
on a different dataset than MACE-MP and intended for molecular systems
rather than periodic materials. It ships in the same `mace-torch` package
as MACE-MP but is loaded via a different entry point.

Models:
    - "small", "medium", "large": MACE-OFF-23 foundation models
"""

def setup(model: str = "medium", device: str = "cuda"):
    from mace.calculators import mace_off
    return mace_off(model=model, device=device, default_dtype="float32")
