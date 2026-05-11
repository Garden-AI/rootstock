# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ase>=3.22",
#     "torch>=2.0",
#     "torch-geometric",
#     "torch-scatter",
#     "torch-sparse",
#     "torch-cluster",
# ]
# Note: torchmd-net must be installed with --no-deps (lightning dep unresolvable via uv)
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
TorchMD-Net environment for Rootstock.

TorchMD-Net provides equivariant transformer architectures for molecular
dynamics. Includes pretrained universal models (e.g., ET-OC20).

Models:
    - Path to a local .ckpt file, or a HuggingFace repo ID.
    - Example pretrained: "tensorfield/ET-OC20" (inorganic) — not universal.

Note: TorchMD-Net's primary use case is organic/biomolecular MD. For
inorganic universal potentials, prefer TensorNet or M3GNet.
"""

CHECKPOINTS = {
    "torchmdnet-et-oc20": "tensorfield/ET-OC20",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a TorchMD-Net calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible calculator.
    """
    import torch
    from torchmdnet.calculators import External

    calc = External(CHECKPOINTS[checkpoint], device=device)
    return calc
