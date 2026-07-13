# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mattersim>=1.1.0",
#     "ase>=3.22",
#     "torch>=2.0",
#     # torch's ROCm wheels depend on this; it lives only on the ROCm
#     # index, so it must be a direct dep for [tool.uv.sources] to route it.
#     "pytorch-triton-rocm",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-rocm" }
# pytorch-triton-rocm = { index = "pytorch-rocm" }
#
# [[tool.uv.index]]
# name = "pytorch-rocm"
# url = "https://download.pytorch.org/whl/rocm6.4"
# explicit = true
# ///
"""MatterSim env (ROCm) - Microsoft's MatterSim universal potential on AMD GPUs."""

CHECKPOINTS = {
    "mattersim-v1-0-0-5m": "MatterSim-v1.0.0-5M",
    "mattersim-v1-0-0-1m": "MatterSim-v1.0.0-1M",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=CHECKPOINTS[checkpoint], device=device)
