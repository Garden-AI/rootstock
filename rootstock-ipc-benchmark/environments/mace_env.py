# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///
"""
MACE environment for Rootstock.

This environment provides access to MACE foundation models for
machine learning interatomic potentials.

Models:
    - "small", "medium", "large": Pre-trained MACE-MP-0 models
    - Path to a .pt file: Custom fine-tuned model
"""


def setup(model: str, device: str = "cuda"):
    """
    Load a MACE calculator.

    Args:
        model: Model identifier. Can be:
            - "small", "medium", "large" for MACE-MP-0 foundation models
            - Path to a .pt file for custom models
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu")

    Returns:
        ASE-compatible calculator
    """
    from mace.calculators import mace_mp

    return mace_mp(model=model, device=device, default_dtype="float32")
