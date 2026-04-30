# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chgnet>=0.4.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
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
  - tensornet_env.py: TensorNet-MatPES (same authors, newer, better)
  - chgnet_env.py: CHGNet (charge-informed, strong on magnetic materials)
  - orb_env.py: Orb v3 (universal, supports periodic systems)

This file loads CHGNet as the practical substitute for M3GNet-PES.
"""


def setup(model: str | None = None, device: str = "cuda"):
    """
    Load CHGNet as a substitute for M3GNet-PES.

    Args:
        model: Optional path to a fine-tuned model. If None, uses CHGNet default.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible CHGNetCalculator.
    """
    from chgnet.model import CHGNetCalculator

    if model:
        return CHGNetCalculator(model_path=model, use_device=device)
    return CHGNetCalculator(use_device=device)
