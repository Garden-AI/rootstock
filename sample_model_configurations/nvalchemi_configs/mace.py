# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "nvalchemi-toolkit[mace]",
#     "ase>=3.22",
# ]
# ///
"""MACE env for batched (nvalchemi) serving.

Hosts MACE foundation checkpoints via nvalchemi's MACEWrapper. The
worker computes COO neighbor lists (declared in the wrapper's
NeighborConfig) unless the client ships them.
"""

CHECKPOINTS = {
    "mace-medium-0b2-batched": "medium-0b2",
}


def setup_batched(checkpoint: str, device: str = "cuda", dtype: str | None = None):
    import torch
    from nvalchemi.models.mace import MACEWrapper

    return MACEWrapper.from_checkpoint(
        CHECKPOINTS[checkpoint],
        device=torch.device(device),
        dtype=getattr(torch, dtype) if dtype else None,
    )
