# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # 2.22 is the first release with uma-s-1p2p1 in the registry.
#     "fairchem-core>=2.22",
#     "ase>=3.22",
#     # Intel XPU (Aurora PVC) torch build. >=2.13: older XPU wheels (e.g. 2.8)
#     # have far slower FP64 kernels -- UMA's first forward took >40 min on
#     # 2.8.0+xpu vs ~2 min on 2.13.0+xpu (PVC tile).
#     "torch>=2.13",
#     # torch's XPU wheels depend on this; it lives only on the XPU index, so it
#     # must be a direct dep for [tool.uv.sources] to route it (the index is
#     # explicit, so transitive-only deps aren't fetched from it).
#     "triton-xpu",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-xpu" }
# triton-xpu = { index = "pytorch-xpu" }
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

UMA is multi-task: setup() requires an explicit `task`
(setup_kwargs={"task": ...} / --kwarg task=...) and errors without one.
Verification picks its own head via VERIFY_KWARGS below.
"""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    # uma-s-1p2 had a known major bug; uma-s-1p2p1 is the fixed,
    # upstream-recommended small model and replaces it here.
    "uma-s-1p2p1": "uma-s-1p2p1",
    "uma-m-1p1": "uma-m-1p1",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "uma:custom": None,
}

UMA_TASKS = ("omat", "omol", "oc20", "odac", "omc")

# Verification-only head selection: smoke-test and a bare `rootstock add`
# verify with these (setup() itself has no default task).
VERIFY_KWARGS = {
    "uma-s-1p1": {"task": "omat"},
    "uma-s-1p2p1": {"task": "omat"},
    "uma-m-1p1": {"task": "omat"},
    "uma:custom": {"task": "omat"},
}


def _require_task(task, checkpoint):
    if task is None:
        raise ValueError(
            f"{checkpoint} is multi-task and has no default head - select one "
            f'with setup_kwargs={{"task": ...}} (or --kwarg task=...): '
            f"one of {', '.join(UMA_TASKS)}"
        )
    if task not in UMA_TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {', '.join(UMA_TASKS)}")
    return task


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


def setup(checkpoint: str, device: str = "xpu", task: str | None = None, **kwargs):
    task = _require_task(task, checkpoint)
    _enable_xpu()
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        CHECKPOINTS[checkpoint], device=device, inference_settings=_fp64_settings()
    )
    return FAIRChemCalculator(predictor, task_name=task, **kwargs)


def setup_from_path(path: str, device: str = "xpu", task: str | None = None, **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads
    # through load_predict_unit, not the registry-name lookup setup() uses.
    _enable_xpu()
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=device, inference_settings=_fp64_settings())
    return FAIRChemCalculator(predictor, task_name=task, **kwargs)
