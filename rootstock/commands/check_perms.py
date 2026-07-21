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

    Priority for the install root: --cluster, then --root or the positional
    root (whose argparse default is $ROOTSTOCK_ROOT), then the config file. The
    cache root always comes from ``resolve_cache_root`` (--cache-root, then the
    install's layout.json declaration, then the cluster registry's legacy
    entry), and equals the install root when the cache isn't split. Returns
    None (after printing an error) on bad input.
    """
    if args.cluster:
        try:
            install_root = get_cluster(args.cluster).root
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return None
    else:
        root = getattr(args, "root_flag", None) or args.root or load_config().root
        if not root:
            print(
                "Error: provide an install root path, --cluster <name>, "
                f"or set {ROOTSTOCK_ROOT_ENV}.",
                file=sys.stderr,
            )
            return None
        install_root = Path(root)

    # Resolve the cache root the same way on both paths: a --cluster whose
    # install declares its cache in layout.json must win over the registry's
    # legacy entry, which is exactly what new split deployments rely on.
    return install_root, resolve_cache_root(install_root, args.cache_root)


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
            "issues": [
                {"path": str(i.path), "problem": i.problem, "ancestor": i.ancestor} for i in issues
            ],
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

    print()
    if any(not i.ancestor for i in issues):
        print(
            "Fix root-level issues with: rootstock setup-perms --group <project-group>"
            " --apply (add --retrofit for existing files)."
        )
    if any(i.ancestor for i in issues):
        print(
            "Ancestor-directory issues are outside the install — fixing them takes"
            " the directory's owner or a facilities ticket."
        )
    return 1
