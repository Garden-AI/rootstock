"""
Cluster registry for Rootstock: a name -> path bootstrap.

The registry exists so users can say ``cluster="perlmutter"`` instead of
remembering an install path. Everything else about an install — most
importantly where its model-weight cache lives — is declared by the install
itself in ``{root}/layout.json`` (see ``rootstock.layout``), because a
registry baked into every release goes stale in pinned clients while the
install's own declaration travels with the install. The per-cluster
``cache_root`` here remains only as a fallback for legacy roots that predate
the declaration.
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
    # sophia/polaris resolve the world-shared serving copy, periodically
    # rsync'd from the /eagle/projects/Rootstock build root (which is not
    # world-readable; admin jobs target it via explicit --root/ROOTSTOCK_ROOT).
    # Polaris mounts the same Eagle filesystem as Sophia, so both ALCF
    # machines share one install.
    "sophia": Cluster(
        root=Path("/lus/eagle/projects/catalyst/world-shared/rootstock"),
    ),
    "polaris": Cluster(
        root=Path("/lus/eagle/projects/catalyst/world-shared/rootstock"),
    ),
    "perlmutter": Cluster(
        root=Path("/global/cfs/cdirs/m5268/rootstock"),
        cache_root=Path("/pscratch/sd/o/oprice/rootstock-cache"),
    ),
    "delta": Cluster(root=Path("/work/hdd/data/rootstock")),
    # ORNL Frontier (AMD MI250X). The install root is a User-Managed Software
    # area, which is read-only on compute nodes -- fine for serving, since
    # workers only read their env. The cache is split onto Lustre because
    # model libraries write weights there at runtime.
    "frontier": Cluster(
        root=Path("/sw/frontier/ums/ums047/rootstock"),
        cache_root=Path("/lustre/orion/ums047/world-shared/rootstock-cache"),
    ),
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


def get_clusters_for_root(root: Path | str) -> list[str]:
    """Reverse lookup: every registered cluster served by an install root.

    More than one entry means a shared install (e.g. sophia/polaris on
    Eagle); empty means the root is unknown to the registry.
    """
    root_str = str(root)
    return [name for name, info in CLUSTER_REGISTRY.items() if str(info.root) == root_str]


def get_cluster_for_root(root: Path | str) -> str | None:
    """Reverse lookup: cluster name for a given install root path.

    Returns None when the root is unknown — or when it is ambiguous because
    several clusters share one install (e.g. sophia/polaris on Eagle), in
    which case picking one by registry order would be a guess. Callers that
    need a definite identity must ask for an explicit cluster name.
    """
    matches = get_clusters_for_root(root)
    if len(matches) == 1:
        return matches[0]
    return None
