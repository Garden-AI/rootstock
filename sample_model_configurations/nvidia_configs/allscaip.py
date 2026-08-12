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
    "allscaip-md-direct-all-omol": "allscaip-md-direct-all-omol",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "allscaip:custom": None,
}


def _fairchem_device(device: str) -> str:
    """Translate an indexed device ("cuda:2") into what fairchem v2 accepts.

    MLIPPredictUnit._setup_device asserts `device in ["cpu", "cuda"]` and then
    resolves the real GPU itself via get_device_for_local_rank(), which returns
    f"cuda:{torch.cuda.current_device()}". So an index has to travel through
    torch's current-device state, not the argument. Verifying several
    checkpoints at once on a multi-GPU node hands each worker "cuda:N" — that
    killed all 8 fairchem-v2 checkpoints on the 2026-08-06 Polaris sync
    (4x A100, VERIFY_JOBS=4), while single-GPU Sophia never hit it.
    """
    if device.startswith("cuda:"):
        import torch

        torch.cuda.set_device(int(device.split(":", 1)[1]))
        return "cuda"
    return device


def setup(checkpoint: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        CHECKPOINTS[checkpoint], device=_fairchem_device(device)
    )
    return FAIRChemCalculator(predictor)


def setup_from_path(path: str, device: str = "cuda"):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads through
    # load_predict_unit, not the registry-name lookup setup() uses.
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=_fairchem_device(device))
    return FAIRChemCalculator(predictor)
