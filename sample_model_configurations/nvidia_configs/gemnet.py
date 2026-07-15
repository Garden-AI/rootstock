# /// script
# requires-python = ">=3.11,<3.12"
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
GemNet environment for Rootstock.

Uses fairchem-core 1.x to access legacy OC20 GemNet checkpoints via
OCPCalculator. These checkpoints are optimized for catalysis systems
(slabs + adsorbates), not general-purpose total-energy materials modeling.

Models:
    - "GemNet-OC-Large-S2EF-OC20-All+MD": GemNet-OC large, default
    - "GemNet-OC-S2EF-OC20-All+MD": GemNet-OC All+MD
    - "GemNet-OC-S2EF-OC20-All": GemNet-OC All
    - "GemNet-dT-S2EF-OC20-All": GemNet-dT / GemNet-T All
"""

CHECKPOINTS = {
    "gemnet-oc-large-s2ef-oc20-all-md": "GemNet-OC-Large-S2EF-OC20-All+MD",
    "gemnet-oc-s2ef-oc20-all-md": "GemNet-OC-S2EF-OC20-All+MD",
    "gemnet-oc-s2ef-oc20-all": "GemNet-OC-S2EF-OC20-All",
    "gemnet-dt-s2ef-oc20-all": "GemNet-dT-S2EF-OC20-All",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a GemNet OC20 calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible OCPCalculator.
    """
    import os
    from fairchem.core import OCPCalculator
    from fairchem.core.models.model_registry import model_name_to_local_file

    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    local_path = model_name_to_local_file(CHECKPOINTS[checkpoint], local_cache=cache_dir)
    return OCPCalculator(checkpoint_path=local_path, cpu=(device == "cpu"))
