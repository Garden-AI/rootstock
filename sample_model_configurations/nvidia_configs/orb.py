# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "orb-models>=0.4.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///
"""Orb env — hosts Orbital Materials' Orb universal potentials."""

CHECKPOINTS = {
    "orb-v2": "orb-v2",
}


def setup(checkpoint: str, device: str = "cuda"):
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    # orb-models exposes one function per checkpoint, e.g. pretrained.orb_v2().
    fn_name = CHECKPOINTS[checkpoint].replace("-", "_")
    load_fn = getattr(pretrained, fn_name)
    orbff = load_fn(device=torch.device(device))
    return ORBCalculator(orbff, device=torch.device(device))
