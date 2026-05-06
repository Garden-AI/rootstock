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

CHECKPOINTS = {
    "mattersim-v1-0-0-5m": "MatterSim-v1.0.0-5M",
    "mattersim-v1-0-0-1m": "MatterSim-v1.0.0-1M",
}


def setup(checkpoint: str, device: str = "cuda"):
    """
    Load a MatterSim calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cpu").

    Returns:
        ASE-compatible calculator.
    """
    from mattersim.forcefield import MatterSimCalculator

    return MatterSimCalculator(load_path=CHECKPOINTS[checkpoint], device=device)
