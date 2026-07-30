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
"""AllScAIP env (ROCm) — FAIRChem scalable attention MLIP trained on OMol25.

Identical to nvidia_configs/allscaip.py except torch resolves from the ROCm
wheel index. fairchem-core v2 is a plain PyPI install (no
torch-geometric/pyg-find-links), and AllScAIP's all-to-all attention lives
in-package with no flash-attention or custom CUDA kernels — so the only ROCm
change is the torch wheel index.

OMol checkpoints expect `charge` and `spin` in `atoms.info`.
"""

CHECKPOINTS = {
    "allscaip-md-conserving-all-omol": "allscaip-md-conserving-all-omol",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "allscaip:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(CHECKPOINTS[checkpoint], device=device)
    return FAIRChemCalculator(predictor)


def setup_from_path(path: str, device: str = "cuda"):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads through
    # load_predict_unit, not the registry-name lookup setup() uses.
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=device)
    return FAIRChemCalculator(predictor)
