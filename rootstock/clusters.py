"""
Cluster configuration for Rootstock.

This module provides mappings from cluster names to install roots and
model-weight cache roots. On most clusters the two coincide; on clusters
where they live on different filesystems (e.g., Perlmutter — CFS for code,
PSCRATCH for the flock-friendly cache) the cluster declares both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cluster:
    """A known cluster's install root and (optional) cache root.

    `cache_root` defaults to `root`. Override it only when the right filesystem
    for code/venvs differs from the right filesystem for the model-weight cache.
    """

    root: Path
    cache_root: Path | None = None

    @property
    def resolved_cache_root(self) -> Path:
        return self.cache_root if self.cache_root is not None else self.root


CLUSTER_REGISTRY: dict[str, Cluster] = {
    "della": Cluster(
        root=Path("/scratch/gpfs/ROSENGROUP/common/rootstock"),
    ),
    "sophia": Cluster(
        root=Path("/eagle/Garden-Ai/rootstock"),
    ),
    "perlmutter": Cluster(
        root=Path("/global/cfs/cdirs/m4845/rootstock"),
        cache_root=Path("/pscratch/sd/w/wengler/rootstock-cache"),
    ),
    "delta": Cluster(
        root=Path("/work/hdd/data/rootstock")
    )
}

def get_cluster(cluster: str) -> Cluster:
    """Look up a known cluster by name."""
    if cluster not in CLUSTER_REGISTRY:
        available = ", ".join(CLUSTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown cluster '{cluster}'. Known clusters: {available}. "
            f"Use root='/path/to/rootstock' for custom locations."
        )
    return CLUSTER_REGISTRY[cluster]


def get_root_for_cluster(cluster: str) -> Path:
    """Get the install root for a known cluster."""
    return get_cluster(cluster).root


def get_cache_root_for_cluster(cluster: str) -> Path:
    """Get the cache root for a known cluster (defaults to install root)."""
    return get_cluster(cluster).resolved_cache_root


def get_cluster_for_root(root: Path | str) -> str | None:
    """Reverse lookup: cluster name for a given install root path."""
    root_str = str(root)
    for cluster, info in CLUSTER_REGISTRY.items():
        if str(info.root) == root_str:
            return cluster
    return None
