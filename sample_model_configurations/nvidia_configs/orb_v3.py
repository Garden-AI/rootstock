# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "orb-models>=0.6.2",
#     "ase>=3.25",
#     "torch>=2.8",
# ]
# ///
"""Orb v3 env — Orbital Materials' Orb v3 universal potentials.

Separate from orb.py because orb-models>=0.5 changed the loader API
(returns a tuple, requires `atoms_adapter` on ORBCalculator, moved calculator
import path) and 0.6.x bumped the Python floor to 3.12 and torch to 2.8.
"""

CHECKPOINTS = {
    "orb-v3-conservative-inf-omat": "orb-v3-conservative-inf-omat",
    "orb-v3-conservative-20-omat":  "orb-v3-conservative-20-omat",
    "orb-v3-direct-inf-omat":       "orb-v3-direct-inf-omat",
    "orb-v3-direct-20-omat":        "orb-v3-direct-20-omat",
    "orb-v3-conservative-inf-mpa":  "orb-v3-conservative-inf-mpa",
    "orb-v3-conservative-20-mpa":   "orb-v3-conservative-20-mpa",
    "orb-v3-direct-inf-mpa":        "orb-v3-direct-inf-mpa",
    "orb-v3-direct-20-mpa":         "orb-v3-direct-20-mpa",
    "orb-v3-conservative-omol":     "orb-v3-conservative-omol",
    "orb-v3-direct-omol":           "orb-v3-direct-omol",
}


def setup(checkpoint: str, device: str = "cuda", precision: str = "float32-high"):
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    fn_name = CHECKPOINTS[checkpoint].replace("-", "_")
    load_fn = getattr(pretrained, fn_name)
    orbff, atoms_adapter = load_fn(device=torch.device(device), precision=precision)
    return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=torch.device(device))
