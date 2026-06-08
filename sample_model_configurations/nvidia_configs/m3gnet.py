# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chgnet>=0.4.0",
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
"""
M3GNet environment for Rootstock — redirected to CHGNet.

M3GNet-MP-2021.2.8-PES (the universal inorganic PES) is no longer accessible
via any modern Python package:
  - matgl 2.x (PyG backend) only has TensorNet on HuggingFace (materialyze)
  - matgl 1.x (DGL backend) pointed to a GitHub URL that was removed
  - The original m3gnet package is archived TensorFlow code
  - materialyze HuggingFace has only M3GNet-Eform (formation energy, not PES)

For universal inorganic PES, use:
  - tensornet.py: TensorNet-MatPES (same authors, newer, better)
  - chgnet.py: CHGNet (charge-informed, strong on magnetic materials)
  - orb.py: Orb v3 (universal, supports periodic systems)

This file loads CHGNet as the practical substitute for M3GNet-PES.
"""

CHECKPOINTS = {
    "m3gnet-mp-2021-2-8-pes": "chgnet-default",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load CHGNet as a substitute for M3GNet-PES.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible CHGNetCalculator.
    """
    from chgnet.model import CHGNetCalculator

    return CHGNetCalculator(use_device=device)
