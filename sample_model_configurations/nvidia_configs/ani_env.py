# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torchani>=2.2",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///
"""
ANI-2x environment for Rootstock.

ANI-2x is a neural network potential for organic molecules containing
H, C, N, O, F, S, Cl. It is not a universal potential — do not use it
for inorganic or periodic systems.

Models:
    - "ANI2x": ANI-2x ensemble (default, 8 networks)
    - "ANI1ccx": ANI-1ccx, trained on CCSD(T)/CBS data (H, C, N, O only)
    - "ANI1x": ANI-1x (H, C, N, O only)
"""

CHECKPOINTS = {
    "ani-2x": "ANI2x",
    "ani-1ccx": "ANI1ccx",
    "ani-1x": "ANI1x",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load an ANI calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible calculator.
    """
    import torchani

    model_map = {
        "ANI2x": torchani.models.ANI2x,
        "ANI1ccx": torchani.models.ANI1ccx,
        "ANI1x": torchani.models.ANI1x,
    }
    model = CHECKPOINTS[checkpoint]

    return model_map[model](periodic_table_index=True).to(device).ase()
