"""``rootstock check-perms`` — standalone shared-install permission check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..clusters import get_cluster
from ..config import load_config
from ..perms import check_permissions
from .common import ROOTSTOCK_ROOT_ENV, resolve_cache_root


def _resolve_roots(args) -> tuple[Path, Path | None] | None:
    """Resolve (install_root, cache_root) to check.

    Priority: --cluster, then the positional root (whose argparse default is
    $ROOTSTOCK_ROOT), then the config file. When the root isn't given via
    --cluster, the cache root comes from --cache-root or a reverse lookup in
    the cluster registry. Returns None (after printing an error) on bad input.
    """
    if args.cluster:
        try:
            cluster = get_cluster(args.cluster)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return None
        return cluster.root, cluster.cache_root

    root = args.root or load_config().root
    if not root:
        print(
            f"Error: provide an install root path, --cluster <name>, or set {ROOTSTOCK_ROOT_ENV}.",
            file=sys.stderr,
        )
        return None

    install_root = Path(root)
    cache_root = Path(args.cache_root) if args.cache_root else resolve_cache_root(install_root)
    return install_root, cache_root


def cmd_check_perms(args) -> int:
    """Read-only permission check of the install root, cache root, and ancestors.

    Exit codes:
        0: No issues found
        1: One or more issues found
        2: Usage error (couldn't resolve roots)
    """
    resolved = _resolve_roots(args)
    if resolved is None:
        return 2
    install_root, cache_root = resolved

    issues = check_permissions(
        install_root,
        cache_root,
        group=args.group,
        include_ancestors=True,
    )

    separate_cache = cache_root is not None and cache_root != install_root

    if args.json:
        payload = {
            "install_root": str(install_root),
            "cache_root": str(cache_root) if separate_cache else str(install_root),
            "ok": not issues,
            "issues": [{"path": str(i.path), "problem": i.problem} for i in issues],
        }
        print(json.dumps(payload, indent=2))
        return 1 if issues else 0

    print(f"Install root: {install_root}")
    if separate_cache:
        print(f"Cache root:   {cache_root}")

    if not issues:
        print("OK: no permission issues found.")
        return 0

    print(f"\nFound {len(issues)} issue(s):")
    for issue in issues:
        print(f"  - {issue.path}: {issue.problem}")
    print(
        "\nRoot-level issues: rootstock setup-perms --group <project-group> --apply"
        " (add --retrofit for existing files)."
        "\nAncestor-directory issues are outside the install — fixing them takes"
        " the directory's owner or a facilities ticket."
        "\nFor a full-tree audit, run scripts/check_world_readable.sh <root>."
    )
    return 1
