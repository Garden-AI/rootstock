# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ase>=3.22",
#     "torch>=2.4.0,<2.10",
#     "mace-torch @ git+https://github.com/ACEsuit/mace.git@main",
#     "graph-longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git",
# ]
# ///
"""MACE-POLAR env — electrostatic/polarizable MACE foundation models (OMol25).

MACE-POLAR-1 is not in the PyPI mace-torch release yet, so this env installs
mace from git main and adds the graph_electrostatics repo, whose distribution
is named graph-longrange and provides the graph_longrange module PolarMACE
needs at runtime. Loaded via mace_polar(), a separate route from the
mace_mp()/mace_off() loaders used by the stable `mace` env.

POLAR checkpoints expect `charge`, `spin`, and `external_field` in atoms.info.
"""

CHECKPOINTS = {
    "mace-polar-1-s": "polar-1-s",
    "mace-polar-1-m": "polar-1-m",
    "mace-polar-1-l": "polar-1-l",
}


def setup(checkpoint: str, device: str = "cuda"):
    from mace.calculators import mace_polar

    return mace_polar(model=CHECKPOINTS[checkpoint], device=device, default_dtype="float32")
