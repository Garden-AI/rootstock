"""``rootstock add`` — idempotent download-or-verify for a checkpoint.

Thin argparse adapter over :func:`rootstock.operations.add_checkpoint`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..environment import CheckpointNotFoundError, list_declared_checkpoints
from ..local_checkpoints import LocalCheckpointError, local_checkpoints_for_root
from ..operations import OperationError, add_checkpoint, parse_setup_kwargs
from .common import get_root_or_exit


def _local_checkpoints_or_empty(root: Path) -> dict:
    """The user's local checkpoints for this root; a corrupt registry warns
    instead of breaking a read-only listing."""
    try:
        return local_checkpoints_for_root(root)
    except LocalCheckpointError as exc:
        print(f"Warning: ignoring local-checkpoint registry: {exc}", file=sys.stderr)
        return {}


def _print_checkpoint_catalog(root: Path) -> int:
    """Print every canonical checkpoint id ``rootstock add`` accepts, grouped
    by hosting env, plus the user's registered local checkpoints. Returns a
    process exit code."""
    declared = list_declared_checkpoints(root)
    if not declared:
        print(
            f"No envs are installed at {root}. "
            f"Run `rootstock install <env-file> --root {root}` first."
        )
        return 0

    print(f"Checkpoints available to add in {root}:")
    for env_name, ckpts in declared.items():
        print(f"  {env_name}:")
        if not ckpts:
            print("    (none)")
            continue
        for ckpt_id in ckpts:
            print(f"    {ckpt_id}")

    local = _local_checkpoints_or_empty(root)
    if local:
        print("  local (this user — already registered, no add needed):")
        for ckpt_id, entry in sorted(local.items()):
            print(f"    {ckpt_id}  (env: {entry.env})")
    return 0


def cmd_add(args) -> int:
    # Downloaded model weights land in the shared cache and must be readable
    # by every user (and writable by co-maintainers), whatever the
    # maintainer's personal umask says.
    os.umask(0o002)

    root = get_root_or_exit(args)

    if getattr(args, "list", False):
        return _print_checkpoint_catalog(root)

    if not args.checkpoint:
        print(
            "Error: a checkpoint id is required (or pass --list to see available ids)",
            file=sys.stderr,
        )
        return 2

    try:
        setup_kwargs = parse_setup_kwargs(args.kwarg)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        add_checkpoint(
            root,
            args.checkpoint,
            device=args.device,
            verify=not args.no_verify,
            push=not args.no_push,
            setup_kwargs=setup_kwargs,
            progress=print,
        )
    except (CheckpointNotFoundError, OperationError) as exc:
        if isinstance(exc, CheckpointNotFoundError) and (
            args.checkpoint in _local_checkpoints_or_empty(root)
        ):
            print(
                f"Error: '{args.checkpoint}' is a locally-registered "
                f"checkpoint — it needs no `rootstock add`. Use it directly, "
                f"or re-verify it with `rootstock smoke-test`.",
                file=sys.stderr,
            )
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0
