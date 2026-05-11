# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
# ]
# ///
"""
MACE-OFF23 environment for Rootstock.

MACE-OFF23 is a transferable force field for organic molecules, distinct from
MACE-MP-0 (which targets inorganic materials). Use this env for molecular
dynamics and geometry optimisation of drug-like and organic systems.

Models:
    - "small":  ~4M params, fastest
    - "medium": ~10M params, balanced (default)
    - "large":  ~28M params, most accurate
"""

CHECKPOINTS = {
    "mace-off23-small": "small",
    "mace-off23-medium": "medium",
    "mace-off23-large": "large",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a MACE-OFF23 calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible calculator.
    """
    from mace.calculators import mace_off

    return mace_off(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
