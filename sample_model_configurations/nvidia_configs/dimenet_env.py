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
DimeNet++ environment for Rootstock.

Uses fairchem-core 1.x to access legacy OC20 DimeNet++ checkpoints via
OCPCalculator. These checkpoints are optimized for catalysis systems
(slabs + adsorbates).

Models:
    - "DimeNet++-S2EF-OC20-All": default
    - "DimeNet++-S2EF-OC20-20M"
    - "DimeNet++-S2EF-OC20-2M"
    - "DimeNet++-S2EF-OC20-200k"
"""


def setup(model: str = "DimeNet++-S2EF-OC20-All", device: str = "cuda"):
    """
    Load a DimeNet++ OC20 calculator.

    Args:
        model: DimeNet++ checkpoint name from the FAIRChem 1.x registry.
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
