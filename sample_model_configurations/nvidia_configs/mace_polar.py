# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
#     # 0.3.16 is the first PyPI release with mace_polar().
#     "mace-torch>=0.3.16",
#     # PolarMACE imports graph_longrange at runtime; the distribution is named
#     # graph-longrange and exists only as this git repo (no PyPI release).
#     "graph-longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git",
# ]
# ///
"""MACE-POLAR env — electrostatic/polarizable MACE foundation models (OMol25).

Kept separate from the stable `mace` env because of the extra git-only
graph-longrange dependency.

POLAR checkpoints expect `charge`, `spin`, and `external_field` in atoms.info.
"""

CHECKPOINTS = {
    "mace-polar-1-s": "polar-1-s",
    "mace-polar-1-m": "polar-1-m",
    "mace-polar-1-l": "polar-1-l",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "mace-polar:custom": None,
}


def setup(checkpoint: str, device: str = "cuda", **kwargs):
    from mace.calculators import mace_polar

    kwargs.setdefault("default_dtype", "float32")
    return mace_polar(model=CHECKPOINTS[checkpoint], device=device, **kwargs)


def setup_from_path(path: str, device: str = "cuda", **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): mace_polar() accepts a
    # weights file directly, keeping the PolarMACE model-type wiring.
    from mace.calculators import mace_polar

    kwargs.setdefault("default_dtype", "float32")
    return mace_polar(model=path, device=device, **kwargs)
