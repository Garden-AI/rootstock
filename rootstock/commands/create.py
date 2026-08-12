"""Create command for scaffolding new environment files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TEMPLATE = '''\
# /// script
# requires-python = ">=3.12"
# dependencies = [
#
# ]
# ///
"""{name} env — TODO: describe."""

# Map canonical checkpoint ids to whatever string the upstream library expects.
# Cluster maintainers run `rootstock add <canonical-id>` and the worker dispatches
# via this dict. Keep the keys aligned with the Almanac's published checkpoint ids.
CHECKPOINTS = {{
    # "TODO-canonical-id": "TODO-upstream-string",
}}


def setup(checkpoint: str, device: str = "cuda", **kwargs):
    """
    Load a calculator for a canonical checkpoint id.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu").
        **kwargs: Forward to the calculator constructor (user escape hatch,
            fed by setup_kwargs= / --kwarg).

    Returns:
        ASE-compatible calculator.
    """
    upstream = CHECKPOINTS[checkpoint]  # noqa: F841 — TODO use this
    raise NotImplementedError("TODO: Implement setup()")
'''


def cmd_new_env(args) -> int:
    """Create a new environment file from template."""
    name = args.name

    # Validate environment name
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
        print(
            f"Error: Invalid environment name '{name}'. "
            "Must start with a letter and contain only letters, numbers, and underscores.",
            file=sys.stderr,
        )
        return 1

    # Bare names — drop any legacy `_env` suffix the user typed.
    env_name = name[:-4] if name.endswith("_env") else name

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path.cwd() / f"{env_name}.py"

    # Check if file already exists
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1

    # Display name (e.g., mace -> MACE)
    display_name = env_name.upper()

    # Write the file
    content = TEMPLATE.format(name=display_name)
    output_path.write_text(content)

    print(f"Created {output_path}")
    print("\nNext steps:")
    print("  1. Add dependencies to the script metadata block")
    print("  2. Fill in CHECKPOINTS with canonical-id → upstream-string mappings")
    print("  3. Implement setup() to dispatch via CHECKPOINTS")
    print(f"  4. Install with: rootstock install {output_path}")

    return 0
