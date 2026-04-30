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


def setup(model: str = "ANI2x", device: str = "cuda"):
    """
    Load an ANI calculator.

    Args:
        model: Model name — "ANI2x", "ANI1ccx", or "ANI1x".
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
    if model not in model_map:
        raise ValueError(f"Unknown ANI model {model!r}. Choose from: {list(model_map)}")

    return model_map[model](periodic_table_index=True).to(device).ase()
