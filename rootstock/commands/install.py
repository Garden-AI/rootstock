"""``rootstock install`` — build environments.

Thin argparse adapter over :func:`rootstock.operations.install_environment`;
process-wide concerns (umask, uv availability, layout checks, permission
warnings) live here, the build itself in the operations layer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..layout import ensure_layout_compatible, write_layout_marker
from ..operations import OperationError, install_environment
from .common import get_root_or_exit, resolve_cache_root, warn_on_permissions


def _install_one(root: Path, source: str, args) -> int:
    """Run one install through the operations layer, mapping to an exit code."""
    try:
        install_environment(
            root,
            source,
            force=args.force,
            upgrade=args.upgrade,
            verbose=args.verbose,
            push=not args.no_push,
            pack=not getattr(args, "no_pack", False),
            progress=print,
        )
    except OperationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_install(args) -> int:
    """
    Install environment(s) from a file, directory, or rebuild by name.

    Accepts:
    - A file path: validates, copies to environments/, and builds
    - A directory path: installs all *.py environment files in the directory
    - An environment name: rebuilds an existing registered environment

    Exit codes:
        0: Success (all environments installed)
        1: One or more installs failed
    """
    from ..environment import check_uv_available

    # Shared installs must be world-readable and group-writable (the recipe in
    # docs/cluster-setup.md). Everything this command creates is derived from
    # public packages, so override any restrictive personal umask for the
    # duration of the build — uv subprocesses inherit it — rather than
    # retrofitting permissions afterwards.
    os.umask(0o002)

    if getattr(args, "models", None):
        print(
            "Error: --models has been removed. Use 'rootstock add' instead:\n"
            "  rootstock add <checkpoint-id>",
            file=sys.stderr,
        )
        return 2

    root = get_root_or_exit(args)
    source = args.source
    source_path = Path(source)

    # Never write into a root laid out by a newer rootstock.
    try:
        ensure_layout_compatible(root)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Check uv is available
    if not check_uv_available():
        print(
            "Error: uv not found in PATH. Install uv: "
            "https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return 1

    # Deliberately not overridable here. Where the weights live is a
    # deployment-time decision (`rootstock init --cache-root`), not a
    # per-build one: install runs once per environment and on every rebuild,
    # so a flag here would let one stray invocation re-point the declaration
    # and scatter checkpoints across two filesystems. Changing it on a
    # populated install means editing {root}/layout.json and moving the
    # weights, which should be deliberate.
    cache_root = resolve_cache_root(root)

    # Surface permission problems before the (slow) build starts.
    if not getattr(args, "no_perm_check", False):
        warn_on_permissions(root, cache_root)

    # Stamp (or backfill, for pre-marker installs) the layout version, and
    # make the install self-describing: declare its cache root. For legacy
    # roots without a declaration this persists the registry's answer, so
    # the install keeps working after pinned clients' registries go stale.
    write_layout_marker(root, cache_root=cache_root)

    # DIRECTORY MODE: install all *.py files
    if source_path.is_dir():
        env_files = sorted(source_path.glob("*.py"))
        if not env_files:
            print(f"Error: No *.py files found in {source_path}", file=sys.stderr)
            return 1

        print(f"Installing {len(env_files)} environment(s) from {source_path}:")
        for f in env_files:
            print(f"  - {f.name}")
        print()

        succeeded = []
        failed = []

        for env_file in env_files:
            print(f"{'=' * 60}")
            print(f"Installing: {env_file.name}")
            print(f"{'=' * 60}")

            if _install_one(root, str(env_file), args) == 0:
                succeeded.append(env_file.stem)
            else:
                failed.append(env_file.stem)

            print()

        # Summary
        print(f"{'=' * 60}")
        print("Summary:")
        print(f"  Succeeded: {len(succeeded)}")
        if succeeded:
            print(f"    {', '.join(succeeded)}")
        print(f"  Failed: {len(failed)}")
        if failed:
            print(f"    {', '.join(failed)}")

        return 1 if failed else 0

    # FILE or NAME MODE: single environment
    return _install_one(root, source, args)
