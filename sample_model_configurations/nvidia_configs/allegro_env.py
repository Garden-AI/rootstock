# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "allegro @ git+https://github.com/mir-group/allegro.git",
#     "nequip>=0.6.0",
#     "ase>=3.22",
#     "torch>=2.0",
#     "torch-geometric",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
Allegro environment for Rootstock.

Allegro is a scalable E(3)-equivariant GNN potential from the NequIP family.
Like NequIP, models are system-specific (trained on a specific element set)
and must be deployed via `nequip-deploy` before use.

Usage: pass a path to a deployed Allegro/NequIP model as `model`.
"""


def setup(model: str, device: str = "cuda"):
    """
    Load an Allegro calculator from a deployed model file.

    Args:
        model: Path to deployed Allegro .pth model (output of `nequip-deploy build`).
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible NequIPCalculator (Allegro uses the NequIP ASE interface).
    """
    from nequip.ase import NequIPCalculator

    return NequIPCalculator.from_deployed_model(model_path=model, device=device)
