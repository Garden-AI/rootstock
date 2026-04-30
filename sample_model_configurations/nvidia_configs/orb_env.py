# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "orb-models>=0.4.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///
"""
Orb environment for Rootstock.

Provides access to Orbital Materials' Orb universal potentials for periodic
and molecular systems.

Models:
    - "orb-v2": Orb v2 universal potential (default)
    - "orb-v3-conservative-inf-omat": Orb v3, conservative forces, infinity
    - "orb-v3-direct-inf-omat": Orb v3, direct forces, infinity
    See orb_models.pretrained for the full checkpoint list.
"""


def setup(model: str = "orb-v2", device: str = "cuda"):
    """
    Load an Orb calculator.

    Args:
        model: Checkpoint name passed to orb_models.pretrained.orb_pretrained().
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible ORBCalculator.
    """
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    # orb-models exposes one function per checkpoint, e.g. pretrained.orb_v2().
    # Map "orb-v2" -> "orb_v2", "orb-v3-conservative-inf-omat" -> "orb_v3_conservative_inf_omat".
    fn_name = model.replace("-", "_")
    load_fn = getattr(pretrained, fn_name)
    orbff = load_fn(device=torch.device(device))
    return ORBCalculator(orbff, device=torch.device(device))
