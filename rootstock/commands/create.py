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
"""
{name} environment for Rootstock.

TODO: Add description of this environment.
"""


def setup(model: str | None = None, device: str = "cuda"):
    """
    Load a calculator.

    Args:
        model: Model identifier or checkpoint name.
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu").

    Returns:
        ASE-compatible calculator.
    """
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

    # Normalize name: ensure it ends with _env
    if not name.endswith("_env"):
        env_name = f"{name}_env"
    else:
        env_name = name

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path.cwd() / f"{env_name}.py"

    # Check if file already exists
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1

    # Generate display name from env_name (e.g., mace_env -> MACE)
    display_name = env_name.replace("_env", "").upper()

    # Write the file
    content = TEMPLATE.format(name=display_name)
    output_path.write_text(content)

    print(f"Created {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Add dependencies to the script metadata block")
    print(f"  2. Implement the setup() function")
    print(f"  3. Install with: rootstock install {output_path}")

    return 0
