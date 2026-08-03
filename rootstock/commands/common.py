"""Common utilities for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

# Re-exported so command modules keep one import site; the resolution itself
# lives in rootstock.config / rootstock.layout so the CLI and the calculator
# share it.
from ..config import ROOTSTOCK_ROOT_ENV, resolve_default_root  # noqa: F401
from ..layout import resolve_cache_root  # noqa: F401


def warn_on_permissions(root: Path, cache_root: Path) -> None:
    """Best-effort, bounded permission check run up front (warn-only).

    A shared install with the wrong perms "works for the maintainer, breaks for
    everyone else" — so we surface it before the slow build, not after. Never
    fails the command; only the root directories are stat'd (no recursion), so
    it's cheap on HPC filesystems.
    """
    from ..perms import check_permissions

    issues = check_permissions(root, cache_root)
    if not issues:
        return

    print(
        "\nWarning: shared-install permissions may be misconfigured:",
        file=sys.stderr,
    )
    for issue in issues:
        print(f"  - {issue.path}: {issue.problem}", file=sys.stderr)
    print(
        "  Fix with: rootstock setup-perms --group <project-group> --apply\n"
        "  (or pass --no-perm-check to silence this)",
        file=sys.stderr,
    )


def resolve_root(args) -> Path:
    """``--root`` (or env/config fallback), with ``--cluster`` as a registry
    bootstrap for admins driving a known cluster by name."""
    if getattr(args, "cluster", None) and not args.root:
        from ..clusters import get_root_for_cluster

        return get_root_for_cluster(args.cluster)
    return get_root_or_exit(args)


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

    root = resolve_default_root()
    if root is not None:
        return root

    print(
        f"Error: --root is required (or set {ROOTSTOCK_ROOT_ENV} environment variable, "
        "or configure root in ~/.config/rootstock/config.toml)",
        file=sys.stderr,
    )
    sys.exit(1)
