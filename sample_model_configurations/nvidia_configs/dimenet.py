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

CHECKPOINTS = {
    "dimenet-plus-plus-s2ef-oc20-all": "DimeNet++-S2EF-OC20-All",
    "dimenet-plus-plus-s2ef-oc20-20m": "DimeNet++-S2EF-OC20-20M",
    "dimenet-plus-plus-s2ef-oc20-2m": "DimeNet++-S2EF-OC20-2M",
    "dimenet-plus-plus-s2ef-oc20-200k": "DimeNet++-S2EF-OC20-200k",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "dimenet:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a DimeNet++ OC20 calculator.

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
