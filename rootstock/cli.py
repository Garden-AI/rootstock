"""
Rootstock CLI.

Commands:
    rootstock build <env_name> --root <path> [--models m1,m2] [--force]
    rootstock build --all --root <path>
    rootstock status --root <path>
    rootstock register <env_file> --root <path>
    rootstock list --root <path>
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_minimum_python_version(requires_python: str) -> str:
    """
    Extract minimum Python version from a requires-python specifier.

    Handles PEP 440 version specifiers like:
        ">=3.10"        -> "3.10"
        ">=3.10,<3.13"  -> "3.10"
        "~=3.10"        -> "3.10"
        ">=3.10.0"      -> "3.10"  (normalized for uv)

    Args:
        requires_python: PEP 440 version specifier string

    Returns:
        Minimum version string suitable for `uv venv --python X.Y`

    Raises:
        ValueError: If no minimum version can be determined
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    spec_set = SpecifierSet(requires_python)

    min_version = None

    for spec in spec_set:
        # Operators that establish a lower bound
        if spec.operator in (">=", "~=", "=="):
            version = Version(spec.version)
            if min_version is None or version < min_version:
                min_version = version
        elif spec.operator == ">":
            # Strict greater-than: we can't determine exact minimum
            # but the version given is a reasonable approximation for uv
            version = Version(spec.version)
            if min_version is None or version < min_version:
                min_version = version

    if min_version is None:
        raise ValueError(
            f"Cannot determine minimum Python version from '{requires_python}'. "
            "Specifier must include >=, ~=, ==, or > constraint."
        )

    # Return major.minor only (uv expects "3.10" not "3.10.0")
    return f"{min_version.major}.{min_version.minor}"


def cmd_build(args) -> int:
    """
    Build a pre-built virtual environment from an environment source file.

    Exit codes:
        0: Success
        1: Build failed
    """
    from .environment import check_uv_available, get_model_cache_env
    from .pep723 import parse_pep723_metadata

    root = Path(args.root)
    env_name = args.env_name

    # Check uv is available
    if not check_uv_available():
        print(
            "Error: uv not found in PATH. Install uv: "
            "https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return 1

    # Find environment source file
    env_source = root / "environments" / f"{env_name}.py"
    if not env_source.exists():
        print(f"Error: Environment source not found: {env_source}", file=sys.stderr)
        available = (
            list((root / "environments").glob("*.py")) if (root / "environments").exists() else []
        )
        if available:
            print(f"Available: {[p.stem for p in available]}", file=sys.stderr)
        return 1

    env_target = root / "envs" / env_name

    # Check if already exists
    if env_target.exists():
        if args.force:
            print(f"Removing existing environment: {env_target}")
            shutil.rmtree(env_target)
        else:
            print(f"Error: Environment already exists: {env_target}", file=sys.stderr)
            print("Use --force to rebuild", file=sys.stderr)
            return 1

    print(f"Building environment: {env_name}")
    print(f"  Source: {env_source}")
    print(f"  Target: {env_target}")

    # Parse PEP 723 metadata
    content = env_source.read_text()
    metadata = parse_pep723_metadata(content)
    if metadata is None:
        print(f"Error: No PEP 723 metadata in {env_source}", file=sys.stderr)
        return 1

    dependencies = metadata.get("dependencies", [])
    requires_python = metadata.get("requires-python", ">=3.10")

    # Extract minimum version properly
    try:
        python_version = extract_minimum_python_version(requires_python)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  Python: {requires_python} -> {python_version}")
    print(f"  Dependencies: {dependencies}")

    # Set up environment for uv commands.
    # UV_PYTHON_INSTALL_DIR ensures Python interpreters are stored in the rootstock
    # root directory, making the entire installation portable across machines/containers.
    python_install_dir = root / ".python"
    python_install_dir.mkdir(parents=True, exist_ok=True)

    # Create virtual environment
    print("\n1. Creating virtual environment...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_python_dir = Path(tmp_dir) / ".python"

        # Download Python to local temp directory
        download_env = os.environ.copy()
        download_env["UV_PYTHON_INSTALL_DIR"] = str(tmp_python_dir)

        result = subprocess.run(
            ["uv", "python", "install", python_version],
            capture_output=True,
            text=True,
            env=download_env,
        )
        if result.returncode != 0:
            print(f"Error downloading Python: {result.stderr}", file=sys.stderr)
            return 1

        # Copy downloaded Python to root directory (if not already there)
        if tmp_python_dir.exists():
            for item in tmp_python_dir.iterdir():
                dest = python_install_dir / item.name
                if not dest.exists():
                    if item.is_dir():
                        print(f"  Copying Python to {dest}")
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

    # Phase 2: Create venv using the Python we just installed
    uv_env = os.environ.copy()
    uv_env["UV_PYTHON_INSTALL_DIR"] = str(python_install_dir)

    result = subprocess.run(
        ["uv", "venv", str(env_target), "--python", python_version],
        capture_output=True,
        text=True,
        env=uv_env,
    )
    if result.returncode != 0:
        print(f"Error creating venv: {result.stderr}", file=sys.stderr)
        return 1

    env_python = env_target / "bin" / "python"

    # Install dependencies using uv pip with --python flag
    print("2. Installing dependencies...")

    if dependencies:
        result = subprocess.run(
            ["uv", "pip", "install", "--python", str(env_python)] + dependencies,
            capture_output=not args.verbose,
            text=True,
            env=uv_env,
        )
        if result.returncode != 0:
            print(
                f"Error installing dependencies: {result.stderr if not args.verbose else ''}",
                file=sys.stderr,
            )
            return 1

    # Install rootstock
    print("3. Installing rootstock...")
    # Find rootstock package path
    import rootstock

    rootstock_path = Path(rootstock.__file__).parent.parent

    result = subprocess.run(
        ["uv", "pip", "install", "--python", str(env_python), str(rootstock_path)],
        capture_output=not args.verbose,
        text=True,
        env=uv_env,
    )
    if result.returncode != 0:
        print(
            f"Error installing rootstock: {result.stderr if not args.verbose else ''}",
            file=sys.stderr,
        )
        return 1

    # Copy environment source file
    print("4. Copying environment source...")
    shutil.copy(env_source, env_target / "env_source.py")

    # Pre-download models if requested
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
        print(f"5. Pre-downloading models: {models}")

        cache_env = get_model_cache_env(root)
        env = {**os.environ, **cache_env}

        for model in models:
            print(f"   Downloading: {model}")
            script = f'''
import sys
sys.path.insert(0, "{env_target}")
from env_source import setup
calc = setup("{model}", "cpu")
print(f"Downloaded model: {model}")
'''
            result = subprocess.run(
                [str(env_python), "-c", script],
                env=env,
                capture_output=not args.verbose,
                text=True,
            )
            if result.returncode != 0:
                print(f"   Warning: Failed to download {model}", file=sys.stderr)
                if args.verbose:
                    print(result.stderr, file=sys.stderr)

    print(f"\nBuilt environment: {env_target}")
    return 0


def cmd_status(args) -> int:
    """Show status of rootstock installation."""
    from .environment import list_built_environments, list_environments

    root = Path(args.root)

    print(f"Rootstock root: {root}")

    # List environment sources
    print("\nEnvironment sources:")
    sources = list_environments(root)
    if not sources:
        print("  (none)")
    else:
        for name, path in sources:
            print(f"  {name}")

    # List built environments
    print("\nBuilt environments:")
    built = list_built_environments(root)
    if not built:
        print("  (none)")
    else:
        for name, path in built:
            # Check if env_source.py exists
            has_source = (path / "env_source.py").exists()
            status = "ready" if has_source else "incomplete"
            print(f"  {name:<20} [{status}]")

    # Show cache sizes
    print("\nCache:")
    cache_dir = root / "cache"
    if cache_dir.exists():
        for subdir in sorted(cache_dir.iterdir()):
            if subdir.is_dir():
                # Get size
                total_size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"  {subdir.name + '/':<20} {size_mb:.1f} MB")
    else:
        print("  (no cache directory)")

    return 0


def cmd_register(args) -> int:
    """Register an environment file to the shared directory."""
    from .pep723 import validate_environment_file

    env_path = Path(args.env_file)
    root = Path(args.root)

    # Validate the file
    print(f"Validating {env_path}...")
    is_valid, error = validate_environment_file(env_path)
    if not is_valid:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # Create environments directory
    env_dir = root / "environments"
    env_dir.mkdir(parents=True, exist_ok=True)

    # Copy file
    dest_path = env_dir / env_path.name
    shutil.copy2(env_path, dest_path)

    print(f"Registered: {env_path.stem} -> {dest_path}")
    return 0


def cmd_list(args) -> int:
    """List registered environments."""
    from .environment import list_built_environments, list_environments

    root = Path(args.root)

    sources = list_environments(root)
    built = list_built_environments(root)
    built_names = {name for name, _ in built}

    if not sources and not built:
        print(f"No environments in {root}")
        return 0

    print(f"Environments in {root}:")
    for name, path in sources:
        status = "built" if name in built_names else "source only"
        print(f"  {name:<20} [{status}]")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="rootstock",
        description="Rootstock MLIP environment manager",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build a pre-built environment",
        description="Build a virtual environment from an environment source file.",
    )
    build_parser.add_argument("env_name", help="Name of environment to build (e.g., mace_env)")
    build_parser.add_argument("--root", required=True, help="Root directory")
    build_parser.add_argument("--models", help="Comma-separated list of models to pre-download")
    build_parser.add_argument("--force", action="store_true", help="Rebuild if exists")
    build_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    build_parser.set_defaults(func=cmd_build)

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of rootstock installation",
        description="Show environment sources, built environments, and cache sizes.",
    )
    status_parser.add_argument("--root", required=True, help="Root directory")
    status_parser.set_defaults(func=cmd_status)

    # register command
    reg_parser = subparsers.add_parser(
        "register",
        help="Register an environment file",
        description="Copy a validated environment file to the shared environments directory.",
    )
    reg_parser.add_argument("env_file", help="Path to environment file")
    reg_parser.add_argument("--root", required=True, help="Root directory")
    reg_parser.set_defaults(func=cmd_register)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List registered environments",
        description="List all environment files in the shared environments directory.",
    )
    list_parser.add_argument("--root", required=True, help="Root directory")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
