# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4.0,<2.5",
#     "ase>=3.22",
#     "huggingface_hub",
#     "matgl",
#     # nvalchemi-toolkit-ops is deliberately absent. It's only an optional
#     # matgl extra (accelerated neighbor lists), and no version can work on
#     # this env's torch pin: 0.3.x registers torch custom ops in modules that
#     # use `from __future__ import annotations`, which torch 2.4's
#     # infer_schema can't parse — ValueError at import, and matgl's optional-
#     # import guard in matgl/ext/ase.py only catches ImportError, so merely
#     # importing PESCalculator crashes — while 0.4+ requires torch>=2.8.
#     # Without it, matgl falls back to its own neighbor list.
#     "pymatgen",
#     "monty",
#     "ruamel.yaml",
#     "scipy",
#     "torch-geometric",
#     "torch-scatter",
#     "torch-sparse",
#     "torch-cluster",
#     "torch-spline-conv",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
#
# [tool.uv.sources]
# # Pinned: this recipe was originally written against matgl 1.0.0, and the
# # unpinned git HEAD silently started building 4.x. setup() below is written
# # for 4.0.3 — bump the tag deliberately, not by rebuild accident.
# matgl = { git = "https://github.com/materialsvirtuallab/matgl.git", tag = "v4.0.3" }
# ///
"""TensorNet env — hosts MatPES TensorNet checkpoints via MatGL."""

CHECKPOINTS = {
    "tensornet-matpes-pbe-2025-2": "materialyze/TensorNet-PES-MatPES-PBE-2025.2",
}


def setup(checkpoint: str, device: str = "cuda", **kwargs):
    from huggingface_hub import snapshot_download

    import matgl
    from matgl.ext.ase import PESCalculator

    # load_model only resolves names against matgl's own manifest; HF models
    # must be downloaded explicitly and passed as a local path.
    local_path = snapshot_download(repo_id=CHECKPOINTS[checkpoint])

    # Move with .to(device), never torch.set_default_device: under matgl 4.x
    # the default-device hack splits the model across devices at load —
    # Potential.__init__ registers data_mean from a constructor-kwarg tensor
    # torch.load restored to cpu, while _eye3 (a persistent=False buffer, not
    # in the state dict) is created fresh on the default device — and
    # forward() then crashes with a cuda/cpu mismatch at
    # `lat @ (self._eye3 + st)` in matgl/apps/pes.py. Module.to() moves
    # params and all buffers coherently, and forward() migrates inputs to the
    # model's device itself.
    pot = matgl.load_model(local_path).to(device)
    return PESCalculator(potential=pot, **kwargs)
