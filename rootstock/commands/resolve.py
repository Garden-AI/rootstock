"""Resolve command for cluster configuration lookup."""

from __future__ import annotations

import json
import sys


def cmd_resolve(args) -> int:
    """Resolve cluster configuration and print as JSON."""
    from ..clusters import get_root_for_cluster

    try:
        root = get_root_for_cluster(args.cluster)
    except ValueError:
        print(f"Error: unknown cluster '{args.cluster}'", file=sys.stderr)
        return 1

    result = {
        "root": str(root),
        "cluster": args.cluster,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Cluster: {args.cluster}")
        print(f"Root:    {root}")
    return 0
