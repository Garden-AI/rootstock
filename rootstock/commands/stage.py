"""``rootstock stage`` — warm checkpoints once, ahead of worker spawns.

The job-prologue command the prewarm ladder called for (#179, subsumed by
node-local staging #180): an sbatch/PBS script stages (or, where staging
isn't configured, page-cache prewarms) the checkpoints it is about to run,
paying the shared-filesystem read exactly once before any worker spawns.
On compute nodes the staged copy / warmth then persists for the job::

    rootstock stage uma-s-1p1 mace-mp-0-medium --cluster delta

Spawns do all of this on their own; the command exists for explicit intent —
warm N checkpoints up front rather than serially on first use.
"""

from __future__ import annotations

import sys

from .. import prewarm
from ..environment import (
    CheckpointNotFoundError,
    get_checkpoint_prewarm_paths,
    resolve_checkpoint,
)
from ..layout import resolve_cache_root
from ..stage import resolve_stage_base, stage_env, stage_weights
from .common import resolve_root


def cmd_stage(args) -> int:
    """
    Stage (or prewarm) each checkpoint's env and weights.

    Exit codes:
        0: Every checkpoint staged or prewarmed
        1: One or more checkpoints failed to resolve or warm
    """
    root = resolve_root(args)
    cluster = getattr(args, "cluster", None)
    cache_root = resolve_cache_root(root)
    base = resolve_stage_base(root)
    if base is None:
        print(
            "Node-local staging is not configured here "
            "(no ROOTSTOCK_STAGE_DIR / layout.json stage_dir); "
            "falling back to a page-cache prewarm pass."
        )

    failures = 0
    for checkpoint_id in args.checkpoints:
        try:
            resolved = resolve_checkpoint(root, checkpoint_id, cluster)
        except CheckpointNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            failures += 1
            continue
        env_name = resolved.env_name
        env_dir = root / "envs" / env_name
        if not (env_dir / "bin" / "python").exists():
            print(f"Error: env '{env_name}' is not built at {env_dir}.", file=sys.stderr)
            failures += 1
            continue

        print(f"{checkpoint_id} (env {env_name}):")
        staged_root = stage_env(root, env_name, base) if base is not None else None
        # Weights mirror only alongside a staged env — that is the only copy
        # spawns will point a worker's caches at (stage_for_spawn returns
        # None outright when the env can't stage).
        staged_weights = None
        if base is not None and staged_root is not None and not resolved.is_custom:
            staged_weights = stage_weights(root, cache_root, env_name, checkpoint_id, base)

        # Whatever didn't stage gets the classic sequential warm instead —
        # through prewarm_from_spec, so the prologue keeps the cgroup
        # working-set warning and ROOTSTOCK_NO_PREWARM semantics.
        spec: dict = {}
        if staged_root is None:
            spec["env_dir"] = str(env_dir)
        if staged_weights is None and not resolved.is_custom:
            try:
                paths, tier = get_checkpoint_prewarm_paths(
                    root, env_name, checkpoint_id, cache_root
                )
            except Exception:
                paths, tier = [], None
            spec["prewarm_paths"] = paths
            if tier is not None:
                spec["prewarm_weights_tier"] = tier
        if spec.get("env_dir") or spec.get("prewarm_paths"):
            prewarm.prewarm_from_spec(spec, log=sys.stdout, label="[Stage]")

    return 1 if failures else 0
