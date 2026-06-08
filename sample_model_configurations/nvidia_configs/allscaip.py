# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fairchem-core>=2.20",
#     "ase>=3.22",
#     "torch>=2.4.0",
# ]
# ///
"""AllScAIP env — FAIRChem scalable attention MLIP trained on OMol25.

allscaip-md-conserving-all-omol is an energy-conserving, all-to-all node
attention model served through fairchem-core's get_predict_unit — the same
API as eSEN. fairchem v2 carries the architecture in-package, so no
flash-attention or custom CUDA kernels are needed.

OMol checkpoints expect `charge` and `spin` in `atoms.info`.
"""

CHECKPOINTS = {
    "allscaip-md-conserving-all-omol": "allscaip-md-conserving-all-omol",
}


def setup(checkpoint: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(CHECKPOINTS[checkpoint], device=device)
    return FAIRChemCalculator(predictor)
