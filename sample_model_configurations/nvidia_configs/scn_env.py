# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "torch>=2.4.0",
#     "fairchem-core>=1.0.0,<2.0.0",
#     "ase>=3.22",
#     "torch-geometric",
#     "torch-scatter",
#     "torch-sparse",
#     "torch-cluster",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
SCN environment for Rootstock.

Uses fairchem-core 1.x to access legacy OC20 SCN checkpoints via
OCPCalculator. These checkpoints are optimized for catalysis systems
(slabs + adsorbates).

Models:
    - "SCN-S2EF-OC20-All+MD": default
    - "SCN-t4-b2-S2EF-OC20-2M"
    - "SCN-S2EF-OC20-2M"
"""


def setup(model: str = "SCN-S2EF-OC20-All+MD", device: str = "cuda"):
    """
    Load an SCN OC20 calculator.

    Args:
        model: SCN checkpoint name from the FAIRChem 1.x registry.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible OCPCalculator.
    """
    import os
    from fairchem.core import OCPCalculator
    from fairchem.core.models.model_registry import model_name_to_local_file

    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    local_path = model_name_to_local_file(model, local_cache=cache_dir)
    return OCPCalculator(checkpoint_path=local_path, cpu=(device == "cpu"))
