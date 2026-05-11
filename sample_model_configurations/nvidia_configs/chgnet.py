# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chgnet>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///
"""CHGNet env — hosts pretrained charge-informed universal potentials."""

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
    from chgnet.model import CHGNet, CHGNetCalculator

    model_name = CHECKPOINTS[checkpoint]
    model = CHGNet.load() if model_name == "chgnet-default" else CHGNet.load(model_name)
    return CHGNetCalculator(model=model, use_device=device)
