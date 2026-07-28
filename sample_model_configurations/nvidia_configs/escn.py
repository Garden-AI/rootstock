# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch>=2.4.0",
#     "fairchem-core>=1.0.0,<2.0.0",
#     "ase>=3.22",
#     # scipy.special.sph_harm was removed in scipy 1.17 and fairchem-core 1.x
#     # still imports it — an uncapped rebuild breaks at import (Delta, 2026-07-18).
#     "scipy<1.17",
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
eSCN environment for Rootstock.

Uses fairchem-core 1.x to access legacy OC20 eSCN checkpoints via
OCPCalculator. These checkpoints are optimized for catalysis systems
(slabs + adsorbates).

Models:
    - "eSCN-L6-M2-Lay12-S2EF-OC20-All+MD": default
    - "eSCN-L6-M3-Lay20-S2EF-OC20-All+MD"
    - "eSCN-L6-M2-Lay12-S2EF-OC20-2M"
    - "eSCN-L4-M2-Lay12-S2EF-OC20-2M"
"""

CHECKPOINTS = {
    "escn-l6-m2-lay12-s2ef-oc20-all-md": "eSCN-L6-M2-Lay12-S2EF-OC20-All+MD",
    "escn-l6-m3-lay20-s2ef-oc20-all-md": "eSCN-L6-M3-Lay20-S2EF-OC20-All+MD",
    "escn-l6-m2-lay12-s2ef-oc20-2m": "eSCN-L6-M2-Lay12-S2EF-OC20-2M",
    "escn-l4-m2-lay12-s2ef-oc20-2m": "eSCN-L4-M2-Lay12-S2EF-OC20-2M",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "escn:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load an eSCN OC20 calculator.

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


def setup_from_path(path: str, device: str = "cuda"):
    # Local checkpoints (`rootstock add-local`): OCPCalculator loads a
    # checkpoint file natively — this is setup() minus the registry download.
    from fairchem.core import OCPCalculator

    return OCPCalculator(checkpoint_path=path, cpu=(device == "cpu"))
