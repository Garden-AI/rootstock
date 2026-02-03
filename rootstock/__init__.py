"""
Rootstock: MLIP calculators with isolated Python environments.

This package provides ASE-compatible calculators that run MLIPs in isolated
subprocess environments, communicating via the i-PI protocol.
"""

from .calculator import RootstockCalculator
from .clusters import CLUSTER_REGISTRY, KNOWN_ENVIRONMENTS, get_root_for_cluster
from .environment import (
    EnvironmentManager,
    get_model_cache_env,
    list_built_environments,
    list_environments,
)
from .pep723 import parse_pep723_metadata, validate_environment_file
from .server import RootstockServer
from .worker import run_worker

__all__ = [
    "RootstockCalculator",
    "RootstockServer",
    "EnvironmentManager",
    "list_environments",
    "list_built_environments",
    "get_model_cache_env",
    "parse_pep723_metadata",
    "validate_environment_file",
    "run_worker",
    "CLUSTER_REGISTRY",
    "KNOWN_ENVIRONMENTS",
    "get_root_for_cluster",
]
__version__ = "0.5.0"
