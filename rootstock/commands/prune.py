"""``rootstock prune`` — the subtractive half of ``sync``.

Thin argparse adapter over :mod:`rootstock.batch` (the prune planner and
delete executor); process-wide concerns (umask, layout checks, permission
warnings) live here, exactly as in ``sync``.

Unlike sync, prune is plan-confirm by default: the failure mode inverts (a
spurious sync build wastes hours; a spurious prune deletes a 14 GB cache
another user's job is warming), so executing requires ``--yes`` or an
interactive confirmation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..batch import (
    execute_prune,
    plan_prune,
    render_prune_plan,
    render_summary,
)
from ..layout import ensure_layout_compatible, write_layout_marker
from ..manifest import ManifestError
from ..operations import OperationError
from .common import get_root_or_exit, resolve_cache_root, warn_on_permissions


def _resolve_root(args) -> Path:
    """``--root`` (or env/config fallback), with ``--cluster`` as a registry
    bootstrap for admins driving a known cluster by name."""
    if getattr(args, "cluster", None) and not args.root:
        from ..clusters import get_root_for_cluster

        return get_root_for_cluster(args.cluster)
    return get_root_or_exit(args)


def _confirm(say) -> bool:
    """Interactive gate for the destructive path; never reached with --yes."""
    if not sys.stdin.isatty():
        print(
            "Error: refusing to delete without --yes when stdin is not a tty "
            "(use --dry-run to inspect the plan)",
            file=sys.stderr,
        )
        return False
    say("")
    try:
        answer = input("Proceed with deletion? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def cmd_prune(args) -> int:
    """
    Plan and execute the removal of undeclared state plus internal garbage.

    Exit codes:
        0: Pruned (or --dry-run, or nothing to prune, or user declined)
        1: One or more delete items failed (re-run to retry)
        2: Usage error
    """
    # Same shared-install stance as install/sync: the manifest and layout
    # marker this rewrites must stay group-writable.
    os.umask(0o002)

    source_dir: Path | None = None
    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.is_dir():
            print(f"Error: {source_dir} is not a directory", file=sys.stderr)
            return 2

    if args.min_age < 0:
        print("Error: --min-age must be >= 0", file=sys.stderr)
        return 2

    root = _resolve_root(args)

    # Never write into a root laid out by a newer rootstock.
    try:
        ensure_layout_compatible(root)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cache_root = resolve_cache_root(root)

    # With --json the machine-readable document owns stdout; everything
    # human-oriented (plan, progress, summary) moves to stderr.
    say = (lambda line: print(line, file=sys.stderr)) if args.json else print

    try:
        plan = plan_prune(
            root,
            source_dir=source_dir,
            envs=args.env,
            checkpoints=args.checkpoint,
            gc_only=args.gc_only,
            deep=args.deep,
            min_age_hours=args.min_age,
            cache_root=cache_root,
        )
    except (OperationError, ManifestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    render_prune_plan(plan, say=say, root=root, cache_root=cache_root)

    if args.dry_run:
        if args.json:
            print(json.dumps({"root": str(root), "plan": plan.to_dict()}, indent=2))
        return 0

    if plan.is_empty:
        if args.json:
            print(json.dumps({"root": str(root), "plan": plan.to_dict(), "results": []}, indent=2))
        return 0

    if not args.yes and not _confirm(say):
        say("Aborted; nothing was deleted.")
        return 0

    # Mutating-command duties: surface permission problems that would turn a
    # half-executed plan into stranded state, and stamp the layout marker.
    if not args.no_perm_check:
        warn_on_permissions(root, cache_root)
    write_layout_marker(root, cache_root=cache_root)

    say("")
    report = execute_prune(
        root,
        plan,
        fail_fast=args.fail_fast,
        push=not args.no_push,
        cache_root=cache_root,
        say=say,
    )

    render_summary(report, say=say)
    if args.json:
        print(
            json.dumps(
                {"root": str(root), "plan": plan.to_dict(), **report.to_dict()},
                indent=2,
            )
        )

    return 1 if report.failed else 0
