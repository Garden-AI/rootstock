# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "nvalchemi-toolkit[aimnet]",
#     "ase>=3.22",
# ]
# ///
"""AIMNet2 env for batched (nvalchemi) serving.

Hosts AIMNet2 via nvalchemi's AIMNet2Wrapper (aimnet package
underneath). ``charge`` is a required model input: batches must carry a
per-system total charge (use 0 for neutral systems).
"""

CHECKPOINTS = {
    "aimnet2-batched": "aimnet2",
}


def setup_batched(checkpoint: str, device: str = "cuda"):
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    return AIMNet2Wrapper.from_checkpoint(CHECKPOINTS[checkpoint], device=device)
