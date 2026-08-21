# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # 2.22 is the first release with uma-s-1p2p1 in the registry.
#     "fairchem-core>=2.22",
#     "ase>=3.22",
#     "torch>=2.4.0",
# ]
# ///
"""UMA env — hosts Meta's UMA foundation model via FAIRChem.

fairchem-core v2 dropped the torch-geometric / pyg-find-links install dance, so
this env is a plain PyPI install. The original uma-s-1 had an extensivity bug
and was removed from the fairchem 2.20 registry — use uma-s-1p1 or uma-s-1p2p1.

UMA is multi-task: every calculation runs one of the model's task heads, and
there is no sensible task-agnostic default — setup() requires an explicit
`task` (setup_kwargs={"task": ...} / --kwarg task=...) and errors without one.
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


def _require_task(task: str | None, checkpoint: str) -> str:
    if task is None:
        raise ValueError(
            f"{checkpoint} is multi-task and has no default head — select one "
            f'with setup_kwargs={{"task": ...}} (or --kwarg task=...): '
            f"one of {', '.join(UMA_TASKS)}"
        )
    if task not in UMA_TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {', '.join(UMA_TASKS)}")
    return task


def setup(checkpoint: str, device: str = "cuda", task: str | None = None, **kwargs):
    task = _require_task(task, checkpoint)
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        CHECKPOINTS[checkpoint], device=_fairchem_device(device)
    )
    return FAIRChemCalculator(predictor, task_name=task, **kwargs)


def setup_from_path(path: str, device: str = "cuda", task: str | None = None, **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads through
    # load_predict_unit, not the registry-name lookup setup() uses. No task
    # validation here — a fine-tune may carry its own task names, or a single
    # task (fairchem itself errors when a multi-task model gets task_name=None).
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    predictor = load_predict_unit(path, device=_fairchem_device(device))
    return FAIRChemCalculator(predictor, task_name=task, **kwargs)
