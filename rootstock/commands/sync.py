"""``rootstock sync`` — converge an install to its declared state.

Thin argparse adapter over :mod:`rootstock.batch` (the planner and phase
executor); process-wide concerns (umask, uv availability, layout checks,
permission warnings) live here, exactly as in ``install``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..batch import PHASES, execute_sync, plan_sync, render_plan, render_summary
from ..environment import CheckpointNotFoundError
from ..layout import ensure_layout_compatible, write_layout_marker
from ..operations import OperationError, resolve_current_cluster
from .common import resolve_cache_root, resolve_root, resolve_source_arg, warn_on_permissions


def _parse_phases(spec: str) -> tuple[str, ...]:
    requested = [phase.strip() for phase in spec.split(",") if phase.strip()]
    unknown = sorted(set(requested) - set(PHASES))
    if unknown:
        raise ValueError(
            f"unknown phase(s): {', '.join(unknown)} (choose from {', '.join(PHASES)})"
        )
    if not requested:
        raise ValueError(f"--phases needs at least one of {', '.join(PHASES)}")
    # Canonical execution order regardless of how the user spelled it.
    return tuple(phase for phase in PHASES if phase in requested)


def cmd_sync(args) -> int:
    """
    Plan and execute the delta between declared and actual install state.

    Exit codes:
        0: Converged (or --dry-run; the plan itself is not a failure)
        1: One or more work items failed (re-run to retry just those)
        2: Usage error
    """
    from ..environment import check_uv_available

    # Same shared-install stance as install/add: everything sync creates is
    # derived from public packages, so group-writable beats a personal umask.
    os.umask(0o002)

    try:
        phases = _parse_phases(args.phases)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    source_dir: Path | None = None
    if args.source_dir:
        try:
            source_dir = resolve_source_arg(args.source_dir)
        except OperationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    root = resolve_root(args)

    # Never write into a root laid out by a newer rootstock.
    try:
        ensure_layout_compatible(root)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if "build" in phases and not check_uv_available():
        print(
            "Error: uv not found in PATH. Install uv: "
            "https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return 1

    cache_root = resolve_cache_root(root)

    # The verification identity: --cluster doubles as the machine name, so
    # on a shared install (sophia/polaris) results land under the right
    # cluster (#208). Resolved strictly only when this run verifies.
    cluster = args.cluster
    if "verify" in phases:
        try:
            cluster = resolve_current_cluster(root, cluster)
        except OperationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    # With --json the machine-readable document owns stdout; everything
    # human-oriented (progress, plan, summary) moves to stderr.
    say = (lambda line: print(line, file=sys.stderr)) if args.json else print

    try:
        plan = plan_sync(
            root,
            source_dir=source_dir,
            envs=args.env,
            checkpoints=args.checkpoint,
            rebuild=args.rebuild,
            phases=phases,
            cluster=cluster,
        )
    except (OperationError, CheckpointNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    render_plan(plan, say=say)

    if args.dry_run:
        if args.json:
            print(json.dumps({"root": str(root), "plan": plan.to_dict()}, indent=2))
        return 0

    if plan.is_empty:
        if args.json:
            print(json.dumps({"root": str(root), "plan": plan.to_dict(), "results": []}, indent=2))
        return 0

    # Surface permission problems before hours of building, and stamp the
    # layout marker — mutating-command duties, skipped on --dry-run above.
    if not args.no_perm_check:
        warn_on_permissions(root, cache_root)
    write_layout_marker(root, cache_root=cache_root)

    say("")
    report = execute_sync(
        root,
        plan,
        jobs=args.jobs,
        verify_jobs=args.verify_jobs,
        verify_timeout=args.verify_timeout,
        device=args.device,
        upgrade=args.upgrade,
        fail_fast=args.fail_fast,
        push=not args.no_push,
        cache_root=cache_root,
        cluster=cluster,
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
