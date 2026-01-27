"""
Cluster configuration for Rootstock.

This module provides mappings from cluster names to root directories
where Rootstock environments and caches are stored.
"""

from __future__ import annotations

from pathlib import Path

# Registry of known clusters and their rootstock root directories
CLUSTER_REGISTRY: dict[str, str] = {
    "modal": "/vol/rootstock",
    "della": "/scratch/gpfs/SHARED/rootstock",
}

# Known environment prefixes for model string parsing
KNOWN_ENVIRONMENTS = ["mace", "chgnet", "orb", "alignn"]


def get_root_for_cluster(cluster: str) -> Path:
    """
    Get the rootstock root directory for a known cluster.

    Args:
        cluster: Name of a known cluster (e.g., "modal", "della")

    Returns:
        Path to the rootstock root directory for that cluster.

    Raises:
        ValueError: If the cluster is not in the registry.
    """
    if cluster not in CLUSTER_REGISTRY:
        available = ", ".join(CLUSTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown cluster '{cluster}'. Known clusters: {available}. "
            f"Use root='/path/to/rootstock' for custom locations."
        )
    return Path(CLUSTER_REGISTRY[cluster])


def parse_model_string(model: str) -> tuple[str, str]:
    """
    Parse model string into (environment_name, model_arg).

    The model string format is:
        "{environment_name}" or "{environment_name}-{model_arg}"

    Examples:
        "mace-medium" -> ("mace", "medium")
        "mace-small" -> ("mace", "small")
        "mace-/path/to/custom.pt" -> ("mace", "/path/to/custom.pt")
        "chgnet" -> ("chgnet", "")
        "chgnet-0.3.0" -> ("chgnet", "0.3.0")

    Args:
        model: Model identifier string.

    Returns:
        Tuple of (environment_name, model_arg).
    """
    for env in KNOWN_ENVIRONMENTS:
        if model == env:
            return (env, "")
        if model.startswith(f"{env}-"):
            model_arg = model[len(env) + 1 :]  # Everything after "env-"
            return (env, model_arg)

    # Unknown format - treat entire string as environment name
    return (model, "")
