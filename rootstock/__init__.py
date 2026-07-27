"""
Rootstock: MLIP calculators with isolated Python environments.

This package provides ASE-compatible calculators that run MLIPs in isolated
subprocess environments, communicating via the i-PI protocol.

The names exported here are the public, semver-protected API. Everything
else (config, manifest, environment management, the worker, the wire
protocol) is internal: importable from its submodule when you need it, but
subject to change without a major version bump.
"""

from importlib.metadata import version as _pkg_version

from .calculator import RootstockCalculator
from .clusters import CLUSTER_REGISTRY, Cluster, get_cluster, get_root_for_cluster
from .environment import CheckpointNotFoundError, CustomWeightsError, list_declared_checkpoints
from .exceptions import RootstockError
from .server import RootstockServer, WorkerDiedError

__all__ = [
    # Calculators and servers
    "RootstockCalculator",
    "RootstockServer",
    "WorkerDiedError",
    "RootstockError",
    # Checkpoint discovery
    "list_declared_checkpoints",
    "CheckpointNotFoundError",
    "CustomWeightsError",
    # Cluster name -> path bootstrap
    "CLUSTER_REGISTRY",
    "Cluster",
    "get_cluster",
    "get_root_for_cluster",
]
__version__ = _pkg_version("rootstock")

# Names removed from the top level in the pre-1.0 API prune, mapped to where
# they live. Gives 0.x users a pointed error instead of a bare AttributeError.
_MOVED = {
    "RootstockClient": "rootstock.client",
    "EnvironmentManager": "rootstock.spawn — its replacement is spawn_in_env",
    "list_environments": "rootstock.environment",
    "list_built_environments": "rootstock.environment",
    "get_model_cache_env": "rootstock.environment",
    "parse_pep723_metadata": "rootstock.pep723",
    "validate_environment_file": "rootstock.pep723",
    "run_worker": "rootstock.worker",
    "get_cache_root_for_cluster": "rootstock.clusters",
    "get_cluster_for_root": "rootstock.clusters",
    "UserConfig": "rootstock.config",
    "load_config": "rootstock.config",
    "save_config": "rootstock.config",
    "Manifest": "rootstock.manifest",
    "load_manifest": "rootstock.manifest",
    "save_manifest": "rootstock.manifest",
}


def __getattr__(name: str):
    if name in _MOVED:
        raise AttributeError(
            f"'{name}' is no longer part of the public rootstock API. "
            f"It is internal; if you depend on it, import it from "
            f"{_MOVED[name]} (no compatibility guarantees)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
