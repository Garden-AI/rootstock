"""
Rootstock: MLIP calculators with isolated Python environments.

This package provides ASE-compatible calculators that run MLIPs in isolated
subprocess environments, communicating via the i-PI protocol.
"""

from .calculator import RootstockCalculator
from .server import RootstockServer

__all__ = ["RootstockCalculator", "RootstockServer"]
__version__ = "0.1.0"
