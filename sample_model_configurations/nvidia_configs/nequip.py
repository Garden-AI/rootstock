# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nequip>=0.6.0",
#     "ase>=3.22",
#     "torch>=2.4.0,<2.5",
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

# NequIP is system-specific: there is no universal pretrained checkpoint to
# ship. Users supply their own model deployed with `nequip-deploy`, whose file
# must be named `*.nequip.pth` or `*.nequip.pt2` (the loader rejects other
# names). Register one with its canonical id → path, e.g.:
#     "my-system-nequip": "/path/to/my_model.nequip.pth",
CHECKPOINTS: dict[str, str] = {}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a NequIP calculator from a deployed model file.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible NequIPCalculator.
    """
    from inspect import signature

    try:
        from nequip.integrations.ase import NequIPCalculator
    except ImportError:
        from nequip.ase import NequIPCalculator

    if hasattr(NequIPCalculator, "from_deployed_model"):
        load_model = NequIPCalculator.from_deployed_model
    elif hasattr(NequIPCalculator, "from_compiled_model"):
        load_model = NequIPCalculator.from_compiled_model
    else:
        raise AttributeError(
            "NequIPCalculator has neither from_deployed_model nor from_compiled_model"
        )

    params = signature(load_model).parameters
    kwargs = {"device": device} if "device" in params else {}
    model = CHECKPOINTS[checkpoint]
    for path_arg in ("model_path", "file_name", "path"):
        if path_arg in params:
            return load_model(**{path_arg: model}, **kwargs)
    return load_model(model, **kwargs)
