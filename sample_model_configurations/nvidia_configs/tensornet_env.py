# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4.0",
#     "ase>=3.22",
#     "huggingface_hub",
#     "matgl",
#     "nvalchemi-toolkit-ops",
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
"""
TensorNet environment for Rootstock.

Provides access to TensorNet models via the MatGL library. Models are hosted
on HuggingFace under the materialyze org and loaded by passing the HF model
ID directly to matgl.load_model().

Models:
    - "materialyze/TensorNet-PES-MatPES-PBE-2025.2": PBE functional (default)
    - "materialyze/TensorNet-PES-MatPES-r2SCAN-2025.2": r2SCAN functional
    - "materialyze/TensorNetDGL-PES-MatPES-PBE-2025.2": DGL-backend variant
"""


def setup(model: str = "materialyze/TensorNet-PES-MatPES-PBE-2025.2", device: str = "cuda"):
    """
    Load a TensorNet/MatGL calculator.

    Args:
        model: HuggingFace model ID (e.g., "materialyze/TensorNet-PES-MatPES-PBE-2025.2").
               Passed directly to matgl.load_model().
        device: PyTorch device string (currently MatGL handles device internally)

    Returns:
        ASE-compatible calculator
    """
    import torch
    torch.set_default_device(device)

    # matgl 1.0.0 imports ExpCellFilter from ase.constraints, but it moved to
    # ase.filters in ASE 3.23. Patch it in before matgl imports.
    import ase.constraints
    if not hasattr(ase.constraints, "ExpCellFilter"):
        from ase.filters import ExpCellFilter
        ase.constraints.ExpCellFilter = ExpCellFilter

    # DGL 2.x imports torchdata.datapipes at init, but torchdata>=0.7 removed
    # datapipes. Stub the minimum needed so DGL imports cleanly. matgl only
    # uses DGL for graph construction, not graphbolt.
    import sys, types

    # DGL 2.x graphbolt imports torchdata submodules removed in torchdata>=0.7.
    # Stub the entire graphbolt subpackage before `import dgl` runs; DGL's
    # __init__ will use our empty stub and skip the real graphbolt initialisation.
    # matgl only uses DGL for graph construction — graphbolt is never called.
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
    local_path = snapshot_download(repo_id=model)
    pot = matgl.load_model(local_path)
    return PESCalculator(potential=pot)