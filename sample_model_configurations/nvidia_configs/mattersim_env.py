# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mattersim>=1.1.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///
"""
MatterSim environment for Rootstock.

Provides access to Microsoft's MatterSim universal potential. MatterSim-v1
covers ~100 elements and supports periodic and non-periodic systems.

Models:
    - "MatterSim-v1.0.0-5M": 5M parameter model (faster, default)
    - "MatterSim-v1.0.0-1M": 1M parameter model (smallest)
"""


def setup(model: str = "MatterSim-v1.0.0-5M", device: str = "cuda"):
    """
    Load a MatterSim calculator.

    Args:
        model: MatterSim checkpoint name.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible calculator.
    """
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=model, device=device)
