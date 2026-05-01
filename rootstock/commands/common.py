"""Common utilities for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

from ..clusters import get_cluster, get_cluster_for_root
from ..config import load_config

# Environment variable for default root directory
ROOTSTOCK_ROOT_ENV = "ROOTSTOCK_ROOT"


def resolve_cache_root(root: Path) -> Path:
    """Reverse-lookup the cache root for a given install root.

    If the install root matches a registered cluster, that cluster's
    ``cache_root`` is returned (defaulting to ``root`` itself when the cluster
    didn't register a separate cache root). For unknown roots, returns ``root``.
    """
    cluster_name = get_cluster_for_root(root)
    if cluster_name is None:
        return root
    return get_cluster(cluster_name).resolved_cache_root


def get_root_or_exit(args) -> Path:
    """
    Get the root directory from args, environment variable, or config file.

    Priority:
    1. --root CLI flag
    2. ROOTSTOCK_ROOT environment variable
    3. root in ~/.config/rootstock/config.toml

    Exits with an error message if none are set.
    """
    if args.root:
        return Path(args.root)

    # Check config file as fallback
    config = load_config()
    if config.root:
        return Path(config.root)

    print(
        f"Error: --root is required (or set {ROOTSTOCK_ROOT_ENV} environment variable, "
        "or configure root in ~/.config/rootstock/config.toml)",
        file=sys.stderr,
    )
    sys.exit(1)
