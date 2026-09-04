"""``rootstock pack`` — pack built envs into single-image archives (#180).

Thin argparse adapter over :func:`rootstock.operations.pack_environments`.
``install`` packs each env it builds; this is the backfill for envs built
before packing existed (or whose install-time pack failed — no zstd on
PATH, say). On clusters with login-node CPU-time caps (Delta), run it in a
batch allocation: zstd across a multi-GB env is exactly the kind of burst
those caps kill.
"""

from __future__ import annotations

import os
import sys

from ..layout import ensure_layout_compatible
from ..manifest import ManifestError
from ..operations import OperationError, pack_environments
from .common import resolve_root


def cmd_pack(args) -> int:
    """
    Pack staging images and record them in the manifest.

    Exit codes:
        0: Every requested image packed (or nothing needed packing)
        1: One or more packs failed
    """
    # Images land on the shared install; same umask stance as install/sync.
    os.umask(0o002)

    root = resolve_root(args)
    try:
        ensure_layout_compatible(root)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.all and args.envs:
        print("Error: name envs or pass --all, not both.", file=sys.stderr)
        return 2

    env_names = args.envs or None
    if args.all:
        from ..environment import list_built_environments

        env_names = [name for name, _ in list_built_environments(root)]
        if not env_names:
            print(f"No built envs at {root}.", file=sys.stderr)
            return 1

    try:
        packed = pack_environments(root, env_names, push=not args.no_push, progress=print)
    except (OperationError, ManifestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if packed:
        print(f"\nPacked {len(packed)} image(s) into {root / 'images'}.")
    return 0
