"""
Rootstock: MLIP calculators with isolated Python environments.

This package provides ASE-compatible calculators that run MLIPs in isolated
subprocess environments, communicating via the i-PI protocol.
"""

from .calculator import RootstockCalculator
from .clusters import CLUSTER_REGISTRY, get_root_for_cluster, parse_model_string
from .environment import EnvironmentManager, list_environments
from .pep723 import parse_pep723_metadata, validate_environment_file
from .server import RootstockServer
from .worker import run_worker

__all__ = [
    "RootstockCalculator",
    "RootstockServer",
    "EnvironmentManager",
    "list_environments",
    "parse_pep723_metadata",
    "validate_environment_file",
    "run_worker",
    "CLUSTER_REGISTRY",
    "get_root_for_cluster",
    "parse_model_string",
]
__version__ = "0.3.0"
