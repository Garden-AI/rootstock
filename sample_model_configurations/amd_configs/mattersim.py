# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mattersim>=1.1.0",
#     "ase>=3.22",
#     "torch>=2.0",
#     # torch's ROCm wheels depend on this; it lives only on the ROCm
#     # index, so it must be a direct dep for [tool.uv.sources] to route it.
#     "pytorch-triton-rocm",
#     # Not imported here. mattersim -> torchmetrics -> torchvision pulls it in
#     # transitively, and torchvision ships compiled ops ABI-locked to one exact
#     # torch build. An `explicit` index is only consulted for packages listed
#     # in `dependencies`, so without this line torchvision silently resolves
#     # from PyPI (a CUDA build!) against a different torch, and the mismatch
#     # surfaces only at `import torchvision` as:
#     #     RuntimeError: operator torchvision::nms does not exist
#     # Leave it unpinned: the index hosts one build per torch release and uv
#     # enforces the torch<->torchvision pairing.
#     "torchvision",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-rocm" }
# pytorch-triton-rocm = { index = "pytorch-rocm" }
# torchvision = { index = "pytorch-rocm" }
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
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "mattersim:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=CHECKPOINTS[checkpoint], device=device)


def setup_from_path(path: str, device: str = "cuda"):
    # Local checkpoints (`rootstock add-local`): load_path accepts a filesystem
    # path as well as a model name, so this is setup() minus the name mapping.
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=path, device=device)
