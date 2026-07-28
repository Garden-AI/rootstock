# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
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
"""MACE env (ROCm) - MACE-MP-0 and MACE-OFF23 on AMD GPUs.

Identical to nvidia_configs/mace.py except torch resolves from the ROCm wheel
index. PyTorch ROCm builds expose AMD GPUs as device="cuda", so setup() is
unchanged. cuEquivariance acceleration is CUDA-only and is not used here  -
MACE falls back to the pure e3nn/torch path.
"""

CHECKPOINTS = {
    "mace-mp-0-small": "small",
    "mace-mp-0-medium": "medium",
    "mace-mp-0-large": "large",
    "mace-off23-small": "off:small",
    "mace-off23-medium": "off:medium",
    "mace-off23-large": "off:large",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "mace-mp:custom": None,
    "mace-off23:custom": None,
}


def setup(checkpoint: str, device: str = "cuda"):
    arg = CHECKPOINTS[checkpoint]
    if arg.startswith("off:"):
        from mace.calculators import mace_off

        return mace_off(model=arg[4:], device=device, default_dtype="float32")
    from mace.calculators import mace_mp

    return mace_mp(model=arg, device=device, default_dtype="float32")


def setup_from_path(path: str, device: str = "cuda"):
    # Custom checkpoints (`:custom` ids with user weights): fine-tunes load through
    # MACECalculator directly — the mp/off dispatch in setup() only exists
    # to pick which pretrained file to download.
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=path, device=device, default_dtype="float32")
