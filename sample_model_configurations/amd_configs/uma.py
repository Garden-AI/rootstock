# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fairchem-core>=2.20",
#     "ase>=3.22",
#     "torch>=2.4.0",
#     # torch's ROCm wheels depend on this; it lives only on the ROCm
#     # index, so it must be a direct dep for [tool.uv.sources] to route it.
#     "pytorch-triton-rocm",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-rocm" }
# pytorch-triton-rocm = { index = "pytorch-rocm" }
#
# [[tool.uv.index]]
# name = "pytorch-rocm"
# url = "https://download.pytorch.org/whl/rocm6.4"
# explicit = true
# ///
"""UMA env (ROCm) - Meta's UMA foundation model via FAIRChem on AMD GPUs.

fairchem-core v2 is a plain PyPI install (no torch-geometric/pyg-find-links),
so the only ROCm change is the torch wheel index. Requires HF_TOKEN for the
gated facebook/UMA checkpoints.
"""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma-s-1p2": "uma-s-1p2",
    "uma-m-1p1": "uma-m-1p1",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "uma:custom": None,
}


def _fairchem_device(device: str) -> str:
    """Translate an indexed device ("cuda:2") into what fairchem v2 accepts.

    MLIPPredictUnit._setup_device asserts `device in ["cpu", "cuda"]` and then
    resolves the real GPU itself via get_device_for_local_rank(), which returns
    f"cuda:{torch.cuda.current_device()}". So an index has to travel through
    torch's current-device state, not the argument. Verifying several
    checkpoints at once on a multi-GPU node hands each worker "cuda:N" — that
    killed all 8 fairchem-v2 checkpoints on the 2026-08-06 Polaris sync
    (4x A100, VERIFY_JOBS=4), while single-GPU Sophia never hit it. ROCm torch
    exposes the same torch.cuda API and "cuda:N" strings, so this applies
    unchanged on AMD.
    """
    if device.startswith("cuda:"):
        import torch

        torch.cuda.set_device(int(device.split(":", 1)[1]))
        return "cuda"
    return device


def setup(checkpoint: str, device: str = "cuda", task: str = "omat"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        CHECKPOINTS[checkpoint], device=_fairchem_device(device)
    )
    return FAIRChemCalculator(predictor, task_name=task)


def setup_from_path(path: str, device: str = "cuda", task: str = "omat"):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads through
    # load_predict_unit, not the registry-name lookup setup() uses.
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=_fairchem_device(device))
    return FAIRChemCalculator(predictor, task_name=task)
