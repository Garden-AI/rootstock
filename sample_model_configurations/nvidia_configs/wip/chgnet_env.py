# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chgnet>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///
"""
CHGNet environment for Rootstock.

This environment provides access to CHGNet, a pretrained universal neural
network potential for charge-informed atomistic modeling.
"""

CHECKPOINTS = {
    "chgnet-default": "chgnet-default",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a CHGNet calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu")

    Returns:
        ASE-compatible calculator
    """
    from chgnet.model import CHGNetCalculator

    return CHGNetCalculator(use_device=device)
