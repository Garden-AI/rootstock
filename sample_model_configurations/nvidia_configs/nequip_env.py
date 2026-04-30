# /// script
# requires-python = ">=3.10"
# dependencies = [
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
NequIP environment for Rootstock.

NequIP (Neural Equivariant Interatomic Potentials) is an E(3)-equivariant
GNN potential. Models must be trained and deployed via `nequip-deploy`;
this env loads a deployed .pth model file.

Usage: pass a path to a deployed NequIP model as `model`.

Note: NequIP is system-specific (not universal). The checkpoint must match
the element set of your system.
"""


def setup(model: str, device: str = "cuda"):
    """
    Load a NequIP calculator from a deployed model file.

    Args:
        model: Path to deployed NequIP .pth model (output of `nequip-deploy build`).
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible NequIPCalculator.
    """
    from nequip.ase import NequIPCalculator

    return NequIPCalculator.from_deployed_model(model_path=model, device=device)
