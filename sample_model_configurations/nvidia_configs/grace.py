# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tensorpotential>=0.6.0",
#     "ase>=3.22",
# ]
# ///
"""GRACE env — hosts GRACE foundation checkpoints via tensorpotential.

GRACE runs on TensorFlow, not torch — tensorpotential pulls
tensorflow[and-cuda] itself (pip-shipped CUDA/cuDNN, so the node's NVIDIA
driver must be compatible). TPCalculator has no device argument: TF grabs
whatever GPU it sees, so device selection happens via CUDA_VISIBLE_DEVICES,
and both it and TF_USE_LEGACY_KERAS must be set before the first TF import.
Weights download ungated from the AMS-ICAMS-RUB HF repo into ~/.cache/grace
under the redirected HOME. The first calculation triggers an XLA compile —
a slow first step is expected.

License: code and all GRACE foundation models are under the Academic
Software License (ASL) — academic/non-commercial use only.
"""

CHECKPOINTS = {
    "grace-2l-smax-omat-large": "GRACE-2L-SMAX-OMAT-large",
    "grace-3l-omat-large-ft-am": "GRACE-3L-OMAT-large-ft-AM",
}


def setup(checkpoint: str, device: str = "cuda"):
    import os

    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]

    from tensorpotential.calculator import grace_fm

    return grace_fm(CHECKPOINTS[checkpoint])
