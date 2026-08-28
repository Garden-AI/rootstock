# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fairchem-core",
#     "ase>=3.26",
#     # Intel XPU (Aurora PVC) torch build. >=2.13: older XPU wheels have far
#     # slower FP64 kernels (see uma.py).
#     "torch>=2.13",
#     # torch's XPU wheels depend on this; it lives only on the XPU index, so it
#     # must be a direct dep for [tool.uv.sources] to route it (the index is
#     # explicit, so transitive-only deps aren't fetched from it).
#     "triton-xpu",
# ]
#
# [tool.uv.sources]
# # Experimental: fairchem-core from a fork whose xpu-support branch adds
# # native Intel-GPU device handling (device="xpu", torch.xpu seeding/cache
# # management, XCCL collectives) -- PyPI fairchem-core accepts only
# # "cpu"/"cuda" device strings. The branch tracks upstream main; switch back
# # to a PyPI fairchem-core once XPU support merges upstream.
# fairchem-core = { git = "https://github.com/abagusetty/fairchem.git", branch = "xpu-support", subdirectory = "packages/fairchem-core" }
# torch = { index = "pytorch-xpu" }
# triton-xpu = { index = "pytorch-xpu" }
#
# [[tool.uv.index]]
# name = "pytorch-xpu"
# url = "https://download.pytorch.org/whl/xpu"
# explicit = true
# ///
"""eSEN env (Intel XPU / Aurora) - FAIRChem eSEN single-task checkpoints.

Same as nvidia_configs/esen.py except (1) torch resolves from the Intel XPU
wheel index, (2) fairchem-core installs from a fork with native XPU support,
and (3) InferenceSettings defaults to float32, so we set
base_precision_dtype=float64 to match the FP64 reference.

Pin one PVC tile with ZE_AFFINITY_MASK in the job (the worker inherits it).
OMol checkpoints expect `charge` and `spin` in `atoms.info`.
"""

CHECKPOINTS = {
    "esen-md-direct-all-omol": "esen-md-direct-all-omol",
    "esen-sm-conserving-all-omol": "esen-sm-conserving-all-omol",
    "esen-sm-direct-all-omol": "esen-sm-direct-all-omol",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "esen:custom": None,
}


def _fairchem_device(device: str) -> str:
    """Translate an indexed device ("xpu:2") into what fairchem accepts.

    MLIPPredictUnit normalizes the requested device to a bare type and resolves
    the actual GPU itself from torch's current-device state
    (torch.xpu.current_device()), so an index has to travel through
    torch.xpu.set_device, not the argument -- the same constraint the CUDA
    configs shim around for multi-GPU verifies.
    """
    if device.startswith("xpu:"):
        import torch

        torch.xpu.set_device(int(device.split(":", 1)[1]))
        return "xpu"
    return device


def _fp64_settings():
    import torch
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    return InferenceSettings(base_precision_dtype=torch.float64, tf32=False)


def setup(checkpoint: str, device: str = "xpu", **kwargs):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        CHECKPOINTS[checkpoint],
        device=_fairchem_device(device),
        inference_settings=_fp64_settings(),
    )
    return FAIRChemCalculator(predictor, **kwargs)


def setup_from_path(path: str, device: str = "xpu", **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads
    # through load_predict_unit, not the registry-name lookup setup() uses.
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(
        path, device=_fairchem_device(device), inference_settings=_fp64_settings()
    )
    return FAIRChemCalculator(predictor, **kwargs)
