# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "torch>=2.4.0",
#     "fairchem-core>=2.0.0",
#     "ase>=3.22",
#     "torch-geometric",
# ]
#
# [tool.uv]
# find-links = ["https://data.pyg.org/whl/torch-2.4.0+cu121.html"]
# ///
"""
eSEN environment for Rootstock.

Provides access to FAIRChem's eSEN single-task checkpoints. Unlike UMA,
eSEN is single-task, so task_name is not passed to FAIRChemCalculator.

Available checkpoints (from fairchem-core's pretrained_mlip registry):
    OMol25:  esen-md-direct-all-omol (default), esen-sm-conserving-all-omol,
             esen-sm-direct-all-omol
    OC25:    esen-sm-conserving-all-oc25, esen-md-direct-all-oc25
    ODAC25:  esen-sm-filtered-odac25, esen-sm-full-odac25

OMol checkpoints expect `charge` and `spin` in `atoms.info`.
"""


def setup(model: str = "esen-md-direct-all-omol", device: str = "cuda"):
    """
    Load an eSEN calculator.

    Args:
        model: Checkpoint name from FAIRChem's pretrained_mlip registry
               (e.g., "esen-md-direct-all-omol", "esen-sm-conserving-all-oc25").
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu").

    Returns:
        ASE-compatible FAIRChemCalculator.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor)
