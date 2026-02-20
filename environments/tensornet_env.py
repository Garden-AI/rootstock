# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.4.0",
#     "ase>=3.22",
#     "matgl @ git+https://github.com/materialsvirtuallab/matgl.git",
#     "nvalchemi-toolkit-ops",
#     "torch-geometric",
#     "torch-scatter",
#     "torch-sparse",
#     "torch-cluster",
#     "torch-spline-conv",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
TensorNet environment for Rootstock.

This environment provides access to TensorNet models via the MatGL library
from the Materials Virtual Lab.

Models:
    - "TensorNet-MatPES-PBE-v2025.1-PES": MatPES PBE functional (default)
    - Other MatGL models as available
"""


def setup(model: str = "TensorNet-MatPES-PBE-v2025.1-PES", device: str = "cuda"):
    """
    Load a TensorNet/MatGL calculator.

    Args:
        model: Model identifier (e.g., "TensorNet-MatPES-PBE-v2025.1-PES").
               Passed directly to matgl.load_model().
        device: PyTorch device string (currently MatGL handles device internally)

    Returns:
        ASE-compatible calculator
    """
    import torch
    torch.set_default_device(device)

    import matgl
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(model)
    return PESCalculator(potential=pot)
