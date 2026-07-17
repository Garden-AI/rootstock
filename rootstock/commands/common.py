"""Common utilities for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

# Re-exported so command modules keep one import site; the resolution itself
# lives in rootstock.config / rootstock.layout so the CLI and the calculator
# share it.
from ..config import ROOTSTOCK_ROOT_ENV, resolve_default_root  # noqa: F401
from ..layout import resolve_cache_root  # noqa: F401


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
