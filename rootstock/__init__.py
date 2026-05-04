"""
Rootstock: MLIP calculators with isolated Python environments.

This package provides ASE-compatible calculators that run MLIPs in isolated
subprocess environments, communicating via the i-PI protocol.
"""

from importlib.metadata import version as _pkg_version

from .calculator import RootstockCalculator
from .client import RootstockClient
from .clusters import (
    CLUSTER_REGISTRY,
    KNOWN_ENVIRONMENTS,
    Cluster,
    get_cache_root_for_cluster,
    get_cluster,
    get_cluster_for_root,
    get_root_for_cluster,
)
from .config import UserConfig, load_config, save_config
from .environment import (
    EnvironmentManager,
    get_model_cache_env,
    list_built_environments,
    list_environments,
)
from .manifest import Manifest, load_manifest, save_manifest
from .pep723 import parse_pep723_metadata, validate_environment_file
from .server import RootstockServer
from .worker import run_worker

__all__ = [
    "RootstockCalculator",
    "RootstockServer",
    "RootstockClient",
    "EnvironmentManager",
    "list_environments",
    "list_built_environments",
    "get_model_cache_env",
    "parse_pep723_metadata",
    "validate_environment_file",
    "run_worker",
    "CLUSTER_REGISTRY",
    "KNOWN_ENVIRONMENTS",
    "Cluster",
    "get_cluster",
    "get_root_for_cluster",
    "get_cache_root_for_cluster",
    "get_cluster_for_root",
    "UserConfig",
    "load_config",
    "save_config",
    "Manifest",
    "load_manifest",
    "save_manifest",
]
__version__ = _pkg_version("rootstock")
