# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mattersim>=1.1.0",
#     "ase>=3.22",
#     "torch>=2.0",
#     # Not imported here. mattersim -> torchmetrics -> torchvision pulls it in
#     # transitively, and torchvision ships compiled ops ABI-locked to one exact
#     # torch build. An `explicit` index is only consulted for packages listed
#     # in `dependencies`, so without this line torchvision silently resolves
#     # from PyPI against a different torch and the mismatch surfaces only at
#     # `import torchvision` as:
#     #     RuntimeError: operator torchvision::nms does not exist
#     # Leave it unpinned: the index hosts one build per torch release and uv
#     # enforces the torch<->torchvision pairing.
#     "torchvision",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""
MatterSim environment for Rootstock.

Provides access to Microsoft's MatterSim universal potential. MatterSim-v1
covers ~100 elements and supports periodic and non-periodic systems.

Models:
    - "MatterSim-v1.0.0-5M": 5M parameter model (faster, default)
    - "MatterSim-v1.0.0-1M": 1M parameter model (smallest)
"""

CHECKPOINTS = {
    "mattersim-v1-0-0-5m": "MatterSim-v1.0.0-5M",
    "mattersim-v1-0-0-1m": "MatterSim-v1.0.0-1M",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "mattersim:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a MatterSim calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible calculator.
    """
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=CHECKPOINTS[checkpoint], device=device)


def setup_from_path(path: str, device: str = "cuda"):
    # Local checkpoints (`rootstock add-local`): load_path accepts a filesystem
    # path as well as a model name, so this is setup() minus the name mapping.
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=path, device=device)
