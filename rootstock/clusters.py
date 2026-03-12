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
    "della": "/scratch/gpfs/ROSENGROUP/common/rootstock",
}

# Known environment families (used for validation)
KNOWN_ENVIRONMENTS = ["mace", "chgnet", "orb", "alignn", "uma", "tensornet"]


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


def get_cluster_for_root(root: Path | str) -> str | None:
    """
    Get cluster name for a given root path (reverse lookup).

    Args:
        root: Path to a rootstock root directory

    Returns:
        Cluster name if root matches a known cluster, None otherwise.
    """
    root_str = str(root)
    for cluster, path in CLUSTER_REGISTRY.items():
        if root_str == path:
            return cluster
    return None
