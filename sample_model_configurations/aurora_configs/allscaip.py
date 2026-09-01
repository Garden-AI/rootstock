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
"""AllScAIP env (Intel XPU / Aurora) - FAIRChem scalable attention MLIP.

Same as nvidia_configs/allscaip.py except (1) torch resolves from the Intel
XPU wheel index and (2) fairchem-core installs from a fork with native XPU
support.

Unlike uma.py/esen.py, this env does NOT force
InferenceSettings(base_precision_dtype=float64): AllScAIP does not support
FP64 inference. Its radius-graph construction creates tensors at the torch
default dtype that crash against a double batch (torch.mm and index_put
dtype mismatches), and even past those, the backbone hard-casts its node
representations to float32 before the output heads
(fairchem models/allscaip/AllScAIP.py), which then mismatches the doubled
head weights. So this env runs the fairchem default float32 -- the same
precision the NVIDIA deployments verify at. If XPU float32 kernels prove
numerically inadequate here (the reason uma.py forces FP64), that will
surface as a verification failure, and FP64 support has to land in
fairchem first.

Pin one PVC tile with ZE_AFFINITY_MASK in the job (the worker inherits it).
OMol checkpoints expect `charge` and `spin` in `atoms.info`.
"""

CHECKPOINTS = {
    "allscaip-md-conserving-all-omol": "allscaip-md-conserving-all-omol",
    "allscaip-md-direct-all-omol": "allscaip-md-direct-all-omol",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "allscaip:custom": None,
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


def setup(checkpoint: str, device: str = "xpu", **kwargs):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        CHECKPOINTS[checkpoint],
        device=_fairchem_device(device),
    )
    return FAIRChemCalculator(predictor, **kwargs)


def setup_from_path(path: str, device: str = "xpu", **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads
    # through load_predict_unit, not the registry-name lookup setup() uses.
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=_fairchem_device(device))
    return FAIRChemCalculator(predictor, **kwargs)
