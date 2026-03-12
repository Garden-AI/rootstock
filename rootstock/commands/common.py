"""Common utilities for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import load_config

# Environment variable for default root directory
ROOTSTOCK_ROOT_ENV = "ROOTSTOCK_ROOT"


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

    # Check config file as fallback
    config = load_config()
    if config.root:
        return Path(config.root)

    print(
        f"Error: --root is required (or set {ROOTSTOCK_ROOT_ENV} environment variable, "
        "or configure root in ~/.config/rootstock/config.toml)",
        file=sys.stderr,
    )
    sys.exit(1)
