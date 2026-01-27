"""
PEP 723 inline script metadata parser.

PEP 723 defines a standard format for embedding metadata in Python scripts
using a TOML block in comments:

    # /// script
    # requires-python = ">=3.10"
    # dependencies = ["numpy", "ase"]
    # ///

Reference: https://peps.python.org/pep-0723/
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# Pattern to match PEP 723 metadata block
# Matches: # /// script\n...# ///
PEP723_PATTERN = re.compile(
    r"^# /// script\s*\n((?:# .*\n)*?)# ///$",
    re.MULTILINE,
)


def parse_pep723_metadata(content: str) -> dict | None:
    """
    Extract PEP 723 metadata from script content.

    Args:
        content: The full content of a Python script

    Returns:
        Dict with parsed TOML metadata, or None if no valid metadata block found.
        Typical keys: 'requires-python', 'dependencies'
    """
    match = PEP723_PATTERN.search(content)
    if match is None:
        return None

    # Extract the TOML content, stripping leading "# " from each line
    toml_lines = []
    for line in match.group(1).splitlines():
        # Remove leading "# " or just "#" for empty lines
        if line.startswith("# "):
            toml_lines.append(line[2:])
        elif line == "#":
            toml_lines.append("")
        else:
            # Shouldn't happen with well-formed metadata, but handle it
            toml_lines.append(line.lstrip("# "))

    toml_content = "\n".join(toml_lines)

    try:
        return tomllib.loads(toml_content)
    except tomllib.TOMLDecodeError:
        return None


def validate_environment_file(path: Path | str) -> tuple[bool, str]:
    """
    Validate that a file is a valid Rootstock environment file.

    A valid environment file must:
    1. Exist and be readable
    2. Have valid PEP 723 metadata with dependencies
    3. Define a setup() function at module level

    Args:
        path: Path to the environment file

    Returns:
        (is_valid, message) tuple. If valid, message is "OK".
        If invalid, message describes the problem.
    """
    path = Path(path)

    # Check file exists
    if not path.exists():
        return False, f"File not found: {path}"

    if not path.is_file():
        return False, f"Not a file: {path}"

    # Read content
    try:
        content = path.read_text()
    except Exception as e:
        return False, f"Cannot read file: {e}"

    # Check PEP 723 metadata
    metadata = parse_pep723_metadata(content)
    if metadata is None:
        return False, "No valid PEP 723 metadata block found"

    if "dependencies" not in metadata:
        return False, "PEP 723 metadata missing 'dependencies' field"

    # Check for setup() function using AST
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError as e:
        return False, f"Syntax error in file: {e}"

    # Look for a function named 'setup' at module level
    setup_found = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "setup":
            setup_found = True
            # Check it has at least 'model' parameter
            args = node.args
            if len(args.args) == 0 and len(args.posonlyargs) == 0:
                return False, "setup() function must accept at least 'model' parameter"
            break

    if not setup_found:
        return False, "No setup() function found at module level"

    return True, "OK"


def get_dependencies(path: Path | str) -> list[str]:
    """
    Get the dependencies list from an environment file.

    Args:
        path: Path to the environment file

    Returns:
        List of dependency strings, or empty list if not found.
    """
    path = Path(path)
    try:
        content = path.read_text()
        metadata = parse_pep723_metadata(content)
        if metadata and "dependencies" in metadata:
            return metadata["dependencies"]
    except Exception:
        pass
    return []


def get_requires_python(path: Path | str) -> str | None:
    """
    Get the requires-python specifier from an environment file.

    Args:
        path: Path to the environment file

    Returns:
        Python version specifier string, or None if not specified.
    """
    path = Path(path)
    try:
        content = path.read_text()
        metadata = parse_pep723_metadata(content)
        if metadata:
            return metadata.get("requires-python")
    except Exception:
        pass
    return None
