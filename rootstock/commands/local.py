"""``rootstock add-local`` / ``rootstock remove-local`` — per-user local
checkpoints (user-supplied weights files).

Unlike ``add``, nothing here touches the shared install: the weights stay
wherever the user put them, and the registration lives in the per-user
registry (see rootstock.local_checkpoints). That's what makes this usable
by non-maintainers against a read-only shared root.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..local_checkpoints import (
    LocalCheckpointError,
    record_local_verification,
    register_local_checkpoint,
    remove_local_checkpoint,
)
from ..operations import parse_setup_kwargs
from ..verify import verify_checkpoint
from .common import get_root_or_exit, resolve_cache_root


def cmd_add_local(args) -> int:
    # Deliberately no umask(0o002) here (contrast cmd_add): the registry is a
    # private per-user file and the weights file already exists — no shared
    # writes happen.
    root = get_root_or_exit(args)
    cache_root = resolve_cache_root(root)
    weights_path = Path(args.path).expanduser().resolve()

    try:
        setup_kwargs = parse_setup_kwargs(args.kwarg)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Hashing {weights_path} ...")
    try:
        entry = register_local_checkpoint(
            root,
            args.id,
            args.env,
            weights_path,
            setup_kwargs=setup_kwargs,
        )
    except LocalCheckpointError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Registered '{args.id}' -> {entry.path}")
    print(f"  env: {entry.env}")
    print(f"  {entry.sha256} ({entry.size} bytes)")

    if args.no_verify:
        print(
            "Skipped verification (--no-verify). Run `rootstock smoke-test` "
            "on a node with the target device to verify."
        )
        return 0

    print(f"Verifying {entry.env}/{args.id} on {args.device}...")
    ok, err = verify_checkpoint(
        root,
        entry.env,
        args.id,
        args.device,
        setup_kwargs=entry.setup_kwargs,
        cache_root=cache_root,
        checkpoint_path=entry.path,
    )
    record_local_verification(
        root,
        args.id,
        ok=ok,
        device=args.device,
        error=None if ok else f"verify: {err}",
    )
    if not ok:
        # The registration is kept (with last_error recorded) so the user can
        # fix the cause and re-verify via smoke-test — same recoverable
        # semantics as `rootstock add`.
        print(f"Error: verification failed: {err}", file=sys.stderr)
        return 1

    print(
        f'Verified. Use it like any checkpoint id: RootstockCalculator(checkpoint="{args.id}", ...)'
    )
    return 0


def cmd_remove_local(args) -> int:
    root = get_root_or_exit(args)
    try:
        entry = remove_local_checkpoint(root, args.checkpoint)
    except LocalCheckpointError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Removed registry entry '{args.checkpoint}' (weights file untouched: {entry.path})")
    return 0
