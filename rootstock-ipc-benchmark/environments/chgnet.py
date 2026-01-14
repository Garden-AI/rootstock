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


def setup(model: str | None = None, device: str = "cuda"):
    """
    Load a CHGNet calculator.

    Args:
        model: Optional path to a fine-tuned model. If None, uses the
               default pre-trained CHGNet model.
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu")

    Returns:
        ASE-compatible calculator
    """
    from chgnet.model import CHGNetCalculator

    if model:
        return CHGNetCalculator(model_path=model, use_device=device)
    return CHGNetCalculator(use_device=device)
