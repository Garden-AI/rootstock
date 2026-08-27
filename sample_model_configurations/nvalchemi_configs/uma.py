# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "nvalchemi-toolkit[uma]",
#     "ase>=3.22",
# ]
# ///
"""UMA env for batched (nvalchemi) serving.

Hosts UMA checkpoints via nvalchemi's UMAWrapper (fairchem
MLIPPredictUnit underneath). fairchem caps torch <2.9 while the engine
side runs newer torch — the version split the worker process exists for.

Multi-head: ``task`` is required (no default head), matching rootstock's
policy for UMA. fairchem builds the radius graph internally, so the
worker skips neighbor-list construction (``compute_neighbors: False``).
"""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
}


def setup_batched(checkpoint: str, device: str = "cuda", *, task: str):
    from nvalchemi.models.uma import UMAWrapper

    wrapper = UMAWrapper.from_checkpoint(CHECKPOINTS[checkpoint], task_name=task, device=device)
    return wrapper, {"compute_neighbors": False}
