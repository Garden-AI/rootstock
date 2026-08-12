# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fairchem-core>=2.20",
#     "ase>=3.22",
#     "torch>=2.4.0",
# ]
# ///
"""UMA env — hosts Meta's UMA foundation model via FAIRChem.

fairchem-core v2 dropped the torch-geometric / pyg-find-links install dance, so
this env is a plain PyPI install. The original uma-s-1 had an extensivity bug
and was removed from the fairchem 2.20 registry — use uma-s-1p1 or uma-s-1p2p1.
"""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    # uma-s-1p2 has a known major bug; uma-s-1p2p1 fixes it and is the
    # upstream-recommended small model. 1p2 stays listed for reproducibility
    # of existing runs.
    "uma-s-1p2": "uma-s-1p2",
    # uma-s-1p2p1 is in fairchem's registry on git main but NOT in any
    # release yet (latest fairchem-core 2.21.0, 2026-06-08, lacks it — the
    # 2026-07-30 sync failed on exactly this). Re-add when the next
    # fairchem-core ships, and bump the dependency floor to that version.
    # "uma-s-1p2p1": "uma-s-1p2p1",
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
    (4x A100, VERIFY_JOBS=4), while single-GPU Sophia never hit it.
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
