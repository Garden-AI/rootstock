# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4.0,<2.5",
#     "ase>=3.22",
#     "huggingface_hub",
#     "matgl",
#     # 0.4+ needs torch>=2.8 at runtime (custom-op registration uses string
#     # annotations infer_schema can't parse on older torch) but only declares
#     # the constraint on its extras, so the resolver won't catch it.
#     "nvalchemi-toolkit-ops<0.4",
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
# matgl = { git = "https://github.com/materialsvirtuallab/matgl.git" }
# ///
"""TensorNet env — hosts MatPES TensorNet checkpoints via MatGL."""

CHECKPOINTS = {
    "tensornet-matpes-pbe-2025-2": "materialyze/TensorNet-PES-MatPES-PBE-2025.2",
}


def setup(checkpoint: str, device: str = "cuda"):
    import torch

    torch.set_default_device(device)

    # matgl 1.0.0 imports ExpCellFilter from ase.constraints, but it moved to
    # ase.filters in ASE 3.23. Patch it in before matgl imports.
    import ase.constraints

    if not hasattr(ase.constraints, "ExpCellFilter"):
        from ase.filters import ExpCellFilter

        ase.constraints.ExpCellFilter = ExpCellFilter

    # DGL 2.x graphbolt imports torchdata submodules removed in torchdata>=0.7.
    # Stub the entire graphbolt subpackage before `import dgl` runs; DGL's
    # __init__ will use our empty stub and skip the real graphbolt initialisation.
    # matgl only uses DGL for graph construction — graphbolt is never called.
    import sys, types

    for _name in [
        "dgl.graphbolt",
        "dgl.graphbolt.base",
        "dgl.graphbolt.dataloader",
        "dgl.graphbolt.feature_fetcher",
        "dgl.graphbolt.minibatch_transformer",
    ]:
        if _name not in sys.modules:
            sys.modules[_name] = types.ModuleType(_name)

    from huggingface_hub import snapshot_download

    import matgl
    from matgl.ext.ase import PESCalculator

    # matgl 1.0.0 load_model only checks the GitHub manifest; HF models must
    # be downloaded explicitly and passed as a local path.
    local_path = snapshot_download(repo_id=CHECKPOINTS[checkpoint])
    pot = matgl.load_model(local_path)
    return PESCalculator(potential=pot)
