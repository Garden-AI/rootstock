# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fairchem-core>=2.20",
#     "ase>=3.22",
#     "torch>=2.4.0",
# ]
# ///
"""UMA env — hosts Meta's UMA foundation model via FAIRChem.

fairchem-core v2 dropped the torch-geometric / pyg-find-links install dance, so
this env is a plain PyPI install. The original uma-s-1 had an extensivity bug
and was removed from the fairchem 2.20 registry — use uma-s-1p1 or uma-s-1p2.
"""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma-s-1p2": "uma-s-1p2",
    "uma-m-1p1": "uma-m-1p1",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "uma:custom": None,
}


def setup(checkpoint: str, device: str = "cuda", task: str = "omat"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(CHECKPOINTS[checkpoint], device=device)
    return FAIRChemCalculator(predictor, task_name=task)


def setup_from_path(path: str, device: str = "cuda", task: str = "omat"):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads through
    # load_predict_unit, not the registry-name lookup setup() uses.
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=device)
    return FAIRChemCalculator(predictor, task_name=task)
