"""``rootstock setup-perms`` — render or apply the shared-install perm recipe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ..clusters import get_cluster
from ..perms import check_permissions, format_command, render_commands
from .common import resolve_cache_root


def _resolve_roots(args) -> tuple[Path, Path] | None:
    """Resolve (install_root, cache_root) from --cluster or the given root.

    Returns None (after printing an error) on bad input. ``cache_root`` equals
    ``install_root`` when there is no separate cache filesystem; ``resolve_cache_root``
    is the single resolution order every entry point shares (explicit override,
    then the install's ``layout.json`` declaration, then the cluster registry's
    legacy entry) — so setup-perms and check-perms can't disagree about which
    paths the recipe covers.
    """
    if args.cluster:
        try:
            install_root = get_cluster(args.cluster).root
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return None
    else:
        root = getattr(args, "root_flag", None) or args.root
        if not root:
            print(
                "Error: provide an install root path or --cluster <name>.",
                file=sys.stderr,
            )
            return None
        install_root = Path(root)

    return install_root, resolve_cache_root(install_root, args.cache_root)


def cmd_setup_perms(args) -> int:
    """Render (dry-run, default) or apply the shared-install permission recipe."""
    resolved = _resolve_roots(args)
    if resolved is None:
        return 1
    install_root, cache_root = resolved

    no_usage_spool = getattr(args, "no_usage_spool", False)
    usage_dir = getattr(args, "usage_dir", None)
    if no_usage_spool and usage_dir:
        print(
            "Error: --usage-dir provisions the usage spool; it can't be "
            "combined with --no-usage-spool.",
            file=sys.stderr,
        )
        return 2

    commands = render_commands(
        install_root,
        cache_root,
        group=args.group,
        retrofit=args.retrofit,
        usage_spool=not no_usage_spool,
        usage_dir=usage_dir,
    )

    if not args.apply:
        # Dry-run (default): print the commands a maintainer (or sysadmin) would
        # run, so they can review or paste them into a script.
        print(f"# Permission recipe for {install_root} (group: {args.group})")
        if cache_root is not None and cache_root != install_root:
            print(f"# Separate cache root: {cache_root}")
        if args.retrofit:
            print("# --retrofit: includes recursive setfacl for existing files")
        for argv in commands:
            print(format_command(argv))
        return 0

    # --apply: confirm, then run each command, stopping at the first failure.
    print(f"About to apply these permissions to {install_root} (group: {args.group}):")
    for argv in commands:
        print(f"  {format_command(argv)}")
    answer = input("Proceed? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return 1

    for argv in commands:
        result = subprocess.run(argv, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: command failed: {format_command(argv)}", file=sys.stderr)
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            return 1

    # Re-check rather than trust: every command can exit 0 and still leave the
    # roots wrong (a filesystem that silently drops setgid, an ACL the fs maps
    # differently). Better to say so here than to have the maintainer discover
    # it from a later check-perms.
    issues = check_permissions(install_root, cache_root, group=args.group)
    if issues:
        print("Permissions applied, but the roots still look wrong:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue.path}: {issue.problem}", file=sys.stderr)
        print(
            "\nThis filesystem may not support part of the recipe; "
            "run 'rootstock check-perms' for the full check.",
            file=sys.stderr,
        )
        return 1

    print("Permissions applied.")
    return 0
