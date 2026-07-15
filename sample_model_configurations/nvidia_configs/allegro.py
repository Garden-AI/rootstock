# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nequip-allegro @ git+https://github.com/mir-group/allegro.git",
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
Allegro environment for Rootstock.

Allegro is a scalable E(3)-equivariant GNN potential from the NequIP family.
Like NequIP, models are system-specific (trained on a specific element set)
and must be deployed via `nequip-deploy` before use.

Usage: pass a path to a deployed Allegro/NequIP model as `model`.
"""

CHECKPOINTS = {
    "allegro-deployed-model": "deployed_allegro.pth",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load an Allegro calculator from a deployed model file.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible NequIPCalculator (Allegro uses the NequIP ASE interface).
    """
    from nequip.ase import NequIPCalculator

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
