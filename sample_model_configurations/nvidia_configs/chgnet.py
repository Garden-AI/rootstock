# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chgnet>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""CHGNet env — hosts pretrained charge-informed universal potentials."""

CHECKPOINTS = {
    "chgnet-default": "chgnet-default",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "chgnet:custom": None,
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


def setup_from_path(path: str, device: str = "cuda"):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads through
    # CHGNet.from_file, not the named-model CHGNet.load() setup() uses.
    from chgnet.model import CHGNet, CHGNetCalculator

    model = CHGNet.from_file(path)
    return CHGNetCalculator(model=model, use_device=device)
