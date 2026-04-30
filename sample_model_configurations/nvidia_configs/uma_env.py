# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "torch>=2.4.0",
#     "fairchem-core>=2.0.0",
#     "ase>=3.22",
#     "torch-geometric",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
UMA (Universal Atomistic Model) environment for Rootstock.

This environment provides access to Meta's UMA foundation model
via the FAIRChem library.

Models:
    - "uma-s-1p1": UMA small model (default)
    - "uma-m-1p1": UMA medium model (more accurate, more memory intensive)

Task:
    This config defaults to the "omat" task for inorganic materials. For
    catalysis, molecules, MOFs, or molecular crystals, copy this env and set
    task_name to "oc20", "omol", "odac", or "omc" respectively.
"""


def setup(model: str = "uma-s-1p1", device: str = "cuda"):
    """
    Load a UMA calculator.

    Args:
        model: Model identifier (e.g., "uma-s-1p1" or "uma-m-1p1"). Passed directly to
               pretrained_mlip.get_predict_unit().
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu")

    Returns:
        ASE-compatible calculator
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name="omat")
