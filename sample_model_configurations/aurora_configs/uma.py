# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fairchem-core>=2.20",
#     "ase>=3.22",
#     # Intel XPU (Aurora PVC) torch build; pytorch-triton-xpu is its runtime dep.
#     "torch>=2.6",
#     "pytorch-triton-xpu",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-xpu" }
# pytorch-triton-xpu = { index = "pytorch-xpu" }
#
# [[tool.uv.index]]
# name = "pytorch-xpu"
# url = "https://download.pytorch.org/whl/xpu"
# explicit = true
# ///
"""UMA env (Intel XPU / Aurora) - Meta's UMA foundation model via FAIRChem.

Same as nvidia_configs/uma.py except (1) torch resolves from the Intel XPU wheel
index, and (2) two XPU-specific fixes in setup():

  * FairChem's MLIPPredictUnit._setup_device asserts device in {cpu, cuda}; we
    monkeypatch it to accept "xpu" (Intel GPU support is not yet upstream).
  * InferenceSettings defaults to float32; we set base_precision_dtype=float64
    so energies/forces match the FP64 reference (fp32 is wrong at ~1e-7).

Pin one PVC tile with ZE_AFFINITY_MASK in the job (the worker inherits it).
Requires HF_TOKEN for the gated facebook/UMA checkpoints (download on a login
node; `rootstock add ... --no-verify`, then verify on a compute node).
"""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma-s-1p2": "uma-s-1p2",
    "uma-m-1p1": "uma-m-1p1",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "uma:custom": None,
}


def _enable_xpu() -> None:
    """Teach FairChem's predict unit to accept device="xpu".

    Upstream MLIPPredictUnit._setup_device allows only "cpu"/"cuda". Idempotent:
    only wraps the original once.
    """
    import fairchem.core.units.mlip_unit.predict as _predict
    import torch

    if getattr(_predict.MLIPPredictUnit._setup_device, "_xpu_patched", False):
        return
    _orig = _predict.MLIPPredictUnit._setup_device

    def _setup_device(self, device):
        if str(device).startswith("xpu"):
            self.device = torch.device(device)
            return
        return _orig(self, device)

    _setup_device._xpu_patched = True
    _predict.MLIPPredictUnit._setup_device = _setup_device


def _fp64_settings():
    import torch
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    return InferenceSettings(base_precision_dtype=torch.float64, tf32=False)


def setup(checkpoint: str, device: str = "xpu", task: str = "omat", **kwargs):
    _enable_xpu()
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        CHECKPOINTS[checkpoint], device=device, inference_settings=_fp64_settings()
    )
    return FAIRChemCalculator(predictor, task_name=task, **kwargs)


def setup_from_path(path: str, device: str = "xpu", task: str = "omat", **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads
    # through load_predict_unit, not the registry-name lookup setup() uses.
    _enable_xpu()
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=device, inference_settings=_fp64_settings())
    return FAIRChemCalculator(predictor, task_name=task, **kwargs)
