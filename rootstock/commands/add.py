"""``rootstock add`` — idempotent download-or-verify for a checkpoint.

Thin argparse adapter over :func:`rootstock.operations.add_checkpoint`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..environment import (
    CheckpointNotFoundError,
    is_custom_checkpoint,
    list_declared_checkpoints,
)
from ..operations import OperationError, add_checkpoint, parse_setup_kwargs
from .common import get_root_or_exit


def _print_checkpoint_catalog(root: Path) -> int:
    """Print every canonical checkpoint id ``rootstock add`` accepts, grouped
    by hosting env. Returns a process exit code."""
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

    if is_custom_checkpoint(args.checkpoint):
        # ':custom' checkpoints have nothing to download or register — the
        # weights are the user's own file, loaded fresh at every use.
        print(
            f"Error: '{args.checkpoint}' is a custom checkpoint — there is "
            f"nothing to add. Use it directly, passing your weights file "
            f"(weights= in Python, --weights on the CLI).",
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
            verify_timeout=args.verify_timeout,
            push=not args.no_push,
            force=args.force,
            cluster=args.cluster,
            setup_kwargs=setup_kwargs,
            progress=print,
        )
    except (CheckpointNotFoundError, OperationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0
