# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch>=2.4.0",
#     "fairchem-core>=2.0.0",
#     "ase>=3.22",
#     "torch-geometric",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""eSEN env — hosts FAIRChem eSEN single-task checkpoints.

OMol checkpoints expect `charge` and `spin` in `atoms.info`.
"""

CHECKPOINTS = {
    "esen-md-direct-all-omol": "esen-md-direct-all-omol",
    "esen-sm-conserving-all-omol": "esen-sm-conserving-all-omol",
    "esen-sm-direct-all-omol": "esen-sm-direct-all-omol",
}


def setup(checkpoint: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(CHECKPOINTS[checkpoint], device=device)
    return FAIRChemCalculator(predictor)


def setup_from_path(path: str, device: str = "cuda"):
    # Local checkpoints (`rootstock add-local`): a weights *file* loads through
    # load_predict_unit, not the registry-name lookup setup() uses.
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=device)
    return FAIRChemCalculator(predictor)
