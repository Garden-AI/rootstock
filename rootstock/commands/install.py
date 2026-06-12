"""Install command for building environments."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import get_root_or_exit, resolve_cache_root

ROOTSTOCK_GITHUB_URL = "https://github.com/Garden-AI/rootstock.git"


def _rootstock_install_spec() -> str:
    """Pin the worker env to the running rootstock build.

    Tagged release ('0.7.2'):       rootstock==0.7.2 (from PyPI)
    Dev build ('...dev0+abc1234'):  rootstock@git+<url>@abc1234 (from GitHub)
    """
    from rootstock import __version__

    if "dev" not in __version__:
        return f"rootstock=={__version__}"

    if "+" not in __version__:
        raise RuntimeError(
            f"Cannot determine rootstock commit from version {__version__!r}. "
            "Worker env needs a tagged release or a dev build with git "
            "context. Commit your changes (and ensure git history is "
            "available) before running `rootstock install`."
        )

    commit_hash = __version__.split("+")[-1]
    return f"rootstock@git+{ROOTSTOCK_GITHUB_URL}@{commit_hash}"


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


def _install_single_environment(
    root: Path,
    source: str,
    force: bool,
    verbose: bool,
    no_push: bool = False,
) -> int:
    """
    Install a single environment from a file path or environment name.

    Returns 0 on success, 1 on failure.
    """
    from ..environment import parse_checkpoints_dict
    from ..pep723 import parse_pep723_metadata, validate_environment_file
    from .manifest import update_and_push_manifest

    source_path = Path(source)

    # Determine mode: file path or environment name
    if source_path.is_file():
        # FILE MODE: validate → copy → build
        env_name = source_path.stem
        env_source = root / "environments" / f"{env_name}.py"

        print(f"Validating {source_path}...")
        is_valid, error = validate_environment_file(source_path)
        if not is_valid:
            print(f"Error: {error}", file=sys.stderr)
            return 1

        try:
            parse_checkpoints_dict(source_path)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        # If the source file is already the registered file (the natural flow
        # on a shared install: drop file into <root>/environments/ then run
        # install), skip the copy and the "already registered" guard.
        already_at_canonical = (
            env_source.exists()
            and source_path.resolve() == env_source.resolve()
        )

        if env_source.exists() and not force and not already_at_canonical:
            print(
                f"Error: Environment '{env_name}' already registered at {env_source}",
                file=sys.stderr,
            )
            print("Use --force to update and rebuild", file=sys.stderr)
            return 1

        # Create environments directory and copy file (unless already there)
        env_dir = root / "environments"
        env_dir.mkdir(parents=True, exist_ok=True)
        if not already_at_canonical:
            shutil.copy2(source_path, env_source)
            print(f"Registered: {source_path} -> {env_source}")

    else:
        # NAME MODE: use existing registered environment
        env_name = source
        env_source = root / "environments" / f"{env_name}.py"

        if not env_source.exists():
            print(f"Error: Environment not found: {env_name}", file=sys.stderr)
            available = (
                list((root / "environments").glob("*.py"))
                if (root / "environments").exists()
                else []
            )
            if available:
                print(f"Available: {[p.stem for p in available]}", file=sys.stderr)
            return 1

    env_target = root / "envs" / env_name

    # Check if venv already exists
    if env_target.exists():
        if force:
            print(f"Removing existing environment: {env_target}")
            shutil.rmtree(env_target)
        else:
            print(f"Error: Environment already built: {env_target}", file=sys.stderr)
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

    # Extract uv-specific config (generic, works for any environment)
    uv_config = metadata.get("tool", {}).get("uv", {})
    find_links = uv_config.get("find-links", [])

    # Extract minimum version properly
    try:
        python_version = extract_minimum_python_version(requires_python)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  Python: {requires_python} -> {python_version}")
    print(f"  Dependencies: {dependencies}")
    if find_links:
        print(f"  Find-links: {find_links}")

    # Ensure home directory exists for model downloads
    home_dir = root / "home"
    home_dir.mkdir(parents=True, exist_ok=True)

    # Set up environment for uv commands.
    # UV_PYTHON_INSTALL_DIR ensures Python interpreters are stored in the rootstock
    # root directory, making the entire installation portable across machines/containers.
    # UV_CACHE_DIR ensures the package cache is shared across users and stored with
    # the rootstock installation rather than in individual home directories.
    python_install_dir = root / ".python"
    python_install_dir.mkdir(parents=True, exist_ok=True)

    uv_cache_dir = root / ".uv-cache"
    uv_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create virtual environment
    print("\n1. Creating virtual environment...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_python_dir = Path(tmp_dir) / ".python"

        # Download Python to local temp directory
        download_env = os.environ.copy()
        download_env["UV_PYTHON_INSTALL_DIR"] = str(tmp_python_dir)
        download_env["UV_CACHE_DIR"] = str(uv_cache_dir)

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
    uv_env["UV_CACHE_DIR"] = str(uv_cache_dir)

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
        pip_cmd = ["uv", "pip", "install", "--python", str(env_python)]
        for link in find_links:
            pip_cmd.extend(["--find-links", link])
        pip_cmd.extend(dependencies)

        result = subprocess.run(
            pip_cmd,
            capture_output=not verbose,
            text=True,
            env=uv_env,
        )
        if result.returncode != 0:
            print(
                f"Error installing dependencies: {result.stderr if not verbose else ''}",
                file=sys.stderr,
            )
            return 1

    # Install rootstock
    print("3. Installing rootstock...")
    rootstock_spec = _rootstock_install_spec()
    print(f"  Installing: {rootstock_spec}")

    result = subprocess.run(
        ["uv", "pip", "install", "--python", str(env_python), rootstock_spec],
        capture_output=not verbose,
        text=True,
        env=uv_env,
    )
    if result.returncode != 0:
        print(
            f"Error installing rootstock: {result.stderr if not verbose else ''}",
            file=sys.stderr,
        )
        return 1

    # Copy environment source file
    print("4. Copying environment source...")
    shutil.copy(env_source, env_target / "env_source.py")

    print(f"\nBuilt environment: {env_target}")

    # Update manifest (quiet=True to avoid cluttering build output)
    update_and_push_manifest(root, quiet=False, push=not no_push)

    return 0


def _warn_on_permissions(root: Path) -> None:
    """Best-effort, bounded permission check run up front (warn-only).

    A shared install with the wrong perms "works for the maintainer, breaks for
    everyone else" — so we surface it before the slow build, not after. Never
    fails the install; only the root directories are stat'd (no recursion), so
    it's cheap on HPC filesystems.
    """
    from ..perms import check_permissions

    issues = check_permissions(root, resolve_cache_root(root))
    if not issues:
        return

    print(
        "\nWarning: shared-install permissions may be misconfigured:",
        file=sys.stderr,
    )
    for issue in issues:
        print(f"  - {issue.path}: {issue.problem}", file=sys.stderr)
    print(
        "  Fix with: rootstock setup-perms --group <project-group> --apply\n"
        "  (or pass --no-perm-check to silence this)",
        file=sys.stderr,
    )


def cmd_install(args) -> int:
    """
    Install environment(s) from a file, directory, or rebuild by name.

    Accepts:
    - A file path: validates, copies to environments/, and builds
    - A directory path: installs all *.py environment files in the directory
    - An environment name: rebuilds an existing registered environment

    Exit codes:
        0: Success (all environments installed)
        1: One or more installs failed
    """
    from ..environment import check_uv_available

    if getattr(args, "models", None):
        print(
            "Error: --models has been removed. Use 'rootstock add' instead:\n"
            "  rootstock add <checkpoint-id>",
            file=sys.stderr,
        )
        return 2

    root = get_root_or_exit(args)
    source = args.source
    source_path = Path(source)

    # Check uv is available
    if not check_uv_available():
        print(
            "Error: uv not found in PATH. Install uv: "
            "https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return 1

    # Surface permission problems before the (slow) build starts.
    if not getattr(args, "no_perm_check", False):
        _warn_on_permissions(root)

    # DIRECTORY MODE: install all *.py files
    if source_path.is_dir():
        env_files = sorted(source_path.glob("*.py"))
        if not env_files:
            print(f"Error: No *.py files found in {source_path}", file=sys.stderr)
            return 1

        print(f"Installing {len(env_files)} environment(s) from {source_path}:")
        for f in env_files:
            print(f"  - {f.name}")
        print()

        succeeded = []
        failed = []

        for env_file in env_files:
            print(f"{'=' * 60}")
            print(f"Installing: {env_file.name}")
            print(f"{'=' * 60}")

            result = _install_single_environment(
                root=root,
                source=str(env_file),
                force=args.force,
                verbose=args.verbose,
                no_push=args.no_push,
            )

            if result == 0:
                succeeded.append(env_file.stem)
            else:
                failed.append(env_file.stem)

            print()

        # Summary
        print(f"{'=' * 60}")
        print("Summary:")
        print(f"  Succeeded: {len(succeeded)}")
        if succeeded:
            print(f"    {', '.join(succeeded)}")
        print(f"  Failed: {len(failed)}")
        if failed:
            print(f"    {', '.join(failed)}")

        return 1 if failed else 0

    # FILE or NAME MODE: single environment
    return _install_single_environment(
        root=root,
        source=source,
        force=args.force,
        verbose=args.verbose,
        no_push=args.no_push,
    )
