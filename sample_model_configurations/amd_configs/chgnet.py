# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "chgnet>=0.3.0",
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
"""CHGNet env (ROCm) - pretrained charge-informed universal potentials on AMD GPUs."""

CHECKPOINTS = {
    "chgnet-default": "chgnet-default",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "chgnet:custom": None,
}


def setup(checkpoint: str, device: str = "cuda", **kwargs):
    from chgnet.model import CHGNet, CHGNetCalculator

    model_name = CHECKPOINTS[checkpoint]
    model = CHGNet.load() if model_name == "chgnet-default" else CHGNet.load(model_name)
    return CHGNetCalculator(model=model, use_device=device, **kwargs)


def setup_from_path(path: str, device: str = "cuda", **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): a weights *file* loads through
    # CHGNet.from_file, not the named-model CHGNet.load() setup() uses.
    from chgnet.model import CHGNet, CHGNetCalculator

    model = CHGNet.from_file(path)
    return CHGNetCalculator(model=model, use_device=device, **kwargs)
