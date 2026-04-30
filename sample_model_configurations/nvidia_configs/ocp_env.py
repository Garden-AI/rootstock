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
OCP (Open Catalyst Project) environment for Rootstock.

Uses fairchem-core 1.x to access the older OC20/OC22 pretrained models via
OCPCalculator. fairchem-core 2.0 dropped support for these architectures
(they are UMA-only from 2.0 onward).

Covers: GemNet-OC, GemNet-T, EquiformerV2, DimeNet++, SCN, eSCN-L6,
        PaiNN, SchNet (all trained on OC20).

These models are optimized for catalysis (slabs + adsorbates). Not universal
potentials — element coverage and stress support vary by checkpoint.

Checkpoint names (pass as the `model` argument):
    GemNet-OC All+MD:   "GemNet-OC-Large-S2EF-OC20-All+MD"
    EquiformerV2 153M:  "EquiformerV2-153M-S2EF-OC20-All+MD"
    eSCN-L6 All+MD:     "eSCN-L6-M2-Lay12-S2EF-OC20-All+MD"
    PaiNN All:          "PaiNN-S2EF-OC20-All"
    SchNet All:         "SchNet-S2EF-OC20-All"
    DimeNet++:          "DimeNet++-S2EF-OC20-All"

See https://fair-chem.github.io/catalysts/models.html for the full list.
"""


def setup(model: str = "GemNet-OC-Large-S2EF-OC20-All+MD", device: str = "cuda"):
    """
    Load an OCP/FAIRChem 1.x calculator.

    Args:
        model: Checkpoint name from fairchem 1.x pretrained model registry.
               model_name_to_local_file downloads weights on first use.
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
