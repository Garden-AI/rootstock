"""Install command for building environments."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..layout import ensure_layout_compatible, write_layout_marker
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
        ">=3.11"        -> "3.11"
        ">=3.11,<3.13"  -> "3.11"
        "~=3.11"        -> "3.11"
        ">=3.11.0"      -> "3.11"  (normalized for uv)

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


def _lockfile_for(env_source: Path) -> Path:
    """Path of the uv lockfile adjacent to an env source file.

    `uv lock --script foo.py` writes `foo.py.lock` next to the script, and
    `uv sync --script foo.py` honors it when present.
    """
    return env_source.parent / (env_source.name + ".lock")


def _install_single_environment(
    root: Path,
    source: str,
    force: bool,
    verbose: bool,
    no_push: bool = False,
    upgrade: bool = False,
) -> int:
    """
    Install a single environment from a file path or environment name.

    Returns 0 on success, 1 on failure.
    """
    from ..environment import parse_checkpoints_dict
    from ..pep723 import validate_environment_file

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
        already_at_canonical = env_source.exists() and source_path.resolve() == env_source.resolve()

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
            # A lockfile shipped alongside the source (e.g. tracked in a git
            # repo next to the env file) is authoritative — carry it along so
            # the build resolves from it instead of from scratch.
            source_lock = _lockfile_for(source_path)
            if source_lock.exists():
                shutil.copy2(source_lock, _lockfile_for(env_source))
                print(f"Registered lockfile: {source_lock} -> {_lockfile_for(env_source)}")

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

    # Check if venv already exists. A --force rebuild does NOT delete the live
    # env here — the new env is built in {root}/.build and swapped in at the
    # end, so worker spawns on a shared install keep working for the whole
    # (slow) build, and a failed rebuild leaves the old env untouched.
    if env_target.exists() and not force:
        print(f"Error: Environment already built: {env_target}", file=sys.stderr)
        print("Use --force to rebuild", file=sys.stderr)
        return 1

    build_root = root / ".build"
    build_root.mkdir(parents=True, exist_ok=True)
    # Clear leftovers from earlier crashed/killed builds of this env.
    for stale in build_root.glob(f"{env_name}.*"):
        shutil.rmtree(stale, ignore_errors=True)
    build_dir = build_root / f"{env_name}.{os.getpid()}"

    print(f"Building environment: {env_name}")
    print(f"  Source: {env_source}")
    print(f"  Target: {env_target}")
    if env_target.exists():
        print(f"  (building in {build_dir}; the live env is swapped out only when done)")

    try:
        return _build_and_swap(
            root=root,
            env_name=env_name,
            env_source=env_source,
            env_target=env_target,
            build_dir=build_dir,
            verbose=verbose,
            no_push=no_push,
            upgrade=upgrade,
        )
    finally:
        # Gone already on success (renamed into place); left behind on any
        # failure path or exception.
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)


def _build_and_swap(
    root: Path,
    env_name: str,
    env_source: Path,
    env_target: Path,
    build_dir: Path,
    verbose: bool,
    no_push: bool,
    upgrade: bool,
) -> int:
    """Build the venv into build_dir, then atomically swap it into env_target."""
    from ..pep723 import parse_pep723_metadata
    from .manifest import update_and_push_manifest

    # Parse PEP 723 metadata
    content = env_source.read_text()
    metadata = parse_pep723_metadata(content)
    if metadata is None:
        print(f"Error: No PEP 723 metadata in {env_source}", file=sys.stderr)
        return 1

    dependencies = metadata.get("dependencies", [])
    requires_python = metadata.get("requires-python", ">=3.11")

    # Extract minimum version properly
    try:
        python_version = extract_minimum_python_version(requires_python)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  Python: {requires_python} -> {python_version}")
    print(f"  Dependencies: {dependencies}")

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

    # --relocatable keeps script shebangs path-independent so the venv built
    # in {root}/.build works unchanged after the rename into envs/. (The
    # interpreter itself lives in {root}/.python and is unaffected either way.)
    result = subprocess.run(
        ["uv", "venv", str(build_dir), "--relocatable", "--python", python_version],
        capture_output=True,
        text=True,
        env=uv_env,
    )
    if result.returncode != 0:
        print(f"Error creating venv: {result.stderr}", file=sys.stderr)
        return 1

    env_python = build_dir / "bin" / "python"

    # Resolve the dependency lockfile. A build is only as reproducible as its
    # resolution: without a lockfile, a rebuild months later re-resolves the
    # env file's version ranges and produces a different env. `uv lock` keeps
    # the pins in an existing lockfile (re-resolving only what a source edit
    # forces), creates the lockfile on first build, and re-resolves everything
    # only with --upgrade.
    lock_path = _lockfile_for(env_source)
    if dependencies:
        print("2. Resolving dependency lockfile...")
        if upgrade:
            print(f"  --upgrade: re-resolving all pins in {lock_path.name}")
        elif lock_path.exists():
            print(f"  Honoring existing lockfile: {lock_path}")
        else:
            print(f"  No lockfile yet; resolving and writing {lock_path}")

        lock_cmd = ["uv", "lock", "--script", str(env_source)]
        if upgrade:
            lock_cmd.append("--upgrade")
        result = subprocess.run(
            lock_cmd,
            capture_output=not verbose,
            text=True,
            env=uv_env,
        )
        if result.returncode != 0:
            # `uv lock` resolves for every platform at once; envs that pull
            # prebuilt wheels from a platform-specific index (e.g. the PyG
            # find-links pages, which ship no macOS wheels) cannot be locked
            # at all. That must not fail the build — it just stays as
            # unreproducible as it was before lockfiles existed.
            print(
                "  Warning: could not resolve a lockfile for this env "
                "(universal resolution failed — common when a find-links "
                "index lacks wheels for some platform)."
                + (
                    " Honoring the existing lockfile as-is."
                    if lock_path.exists()
                    else " Building without one; rebuilds of this env will re-resolve."
                ),
                file=sys.stderr,
            )
            if not verbose and result.stderr:
                print(f"  uv lock said: {result.stderr.strip().splitlines()[-1]}", file=sys.stderr)
    else:
        print("2. Resolving dependency lockfile... (no dependencies, skipped)")

    # Install dependencies with `uv sync --script` so the env source's full
    # PEP 723 uv config is honored — not just `dependencies`, but
    # `[tool.uv.sources]`, `[[tool.uv.index]]`, and `[tool.uv]` find-links.
    # The `uv pip` interface silently ignores sources/index pins. `--active` +
    # VIRTUAL_ENV targets the venv we just created. Whenever a lockfile
    # exists, `--frozen` installs exactly its pins — sync must never
    # re-resolve on its own. With no lockfile (unlockable env), sync falls
    # back to a plain current-platform resolution.
    print("3. Installing dependencies...")

    if dependencies:
        sync_cmd = ["uv", "sync", "--script", str(env_source), "--active"]
        if lock_path.exists():
            sync_cmd.append("--frozen")
        sync_env = dict(uv_env)
        sync_env["VIRTUAL_ENV"] = str(build_dir)
        result = subprocess.run(
            sync_cmd,
            capture_output=not verbose,
            text=True,
            env=sync_env,
        )
        if result.returncode != 0:
            print(
                f"Error installing dependencies: {result.stderr if not verbose else ''}",
                file=sys.stderr,
            )
            return 1

    # Install rootstock
    print("4. Installing rootstock...")
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

    # Copy environment source file (and its lockfile, so the built env records
    # exactly what it was resolved from)
    print("5. Copying environment source...")
    shutil.copy(env_source, build_dir / "env_source.py")
    if dependencies and lock_path.exists():
        shutil.copy(lock_path, build_dir / "env_source.py.lock")

    # Swap the finished build into place. Two renames (same filesystem, so
    # each is atomic) shrink the unavailable window from the whole build to
    # microseconds; a failure before this point never touched the live env.
    print("6. Swapping new environment into place...")
    if env_target.exists():
        displaced = build_dir.parent / f"{env_name}.old.{os.getpid()}"
        env_target.rename(displaced)
        try:
            build_dir.rename(env_target)
        except OSError:
            displaced.rename(env_target)  # put the old env back
            raise
        shutil.rmtree(displaced, ignore_errors=True)
    else:
        env_target.parent.mkdir(parents=True, exist_ok=True)
        build_dir.rename(env_target)

    env_python = env_target / "bin" / "python"

    print("7. Pre-compiling bytecode...")
    _precompile_environment(env_python, env_target)

    print(f"\nBuilt environment: {env_target}")

    # Update manifest (quiet=True to avoid cluttering build output).
    # built_env stamps this env's built_at to now — the one moment the true
    # build time is known.
    update_and_push_manifest(root, quiet=False, push=not no_push, built_env=env_name)

    return 0


def _precompile_environment(env_python: Path, env_target: Path) -> None:
    """Byte-compile the whole venv so shared-install users never need to.

    Without this, the first import by each user tries to write ``__pycache__``
    into the shared (read-only to them) venv, silently fails, and recompiles
    in memory on every run — a per-import perf tax on everyone but the
    maintainer.

    Warn-only: large site-packages trees routinely contain files that don't
    byte-compile (vendored test fixtures, py2 leftovers). Those fail at import
    time too, so nothing real is importing them; they don't invalidate the
    build.
    """
    env = os.environ.copy()
    # .pyc files must land in-tree, next to their sources — not in the
    # maintainer's per-user redirect, where other users can't see them.
    env.pop("PYTHONPYCACHEPREFIX", None)

    result = subprocess.run(
        [str(env_python), "-m", "compileall", "-q", "-j", "0", str(env_target)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        # compileall reports errors on stdout; -q suppresses everything else.
        output = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
        shown = "\n".join(f"    {line}" for line in output[:10])
        print(
            "  Warning: some files did not byte-compile (normal for vendored/"
            "py2 files; they would fail at import time anyway):\n" + shown
        )
        if len(output) > 10:
            print(f"    ... and {len(output) - 10} more lines")


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

    # Shared installs must be world-readable and group-writable (the recipe in
    # docs/cluster-setup.md). Everything this command creates is derived from
    # public packages, so override any restrictive personal umask for the
    # duration of the build — uv subprocesses inherit it — rather than
    # retrofitting permissions afterwards.
    os.umask(0o002)

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

    # Never write into a root laid out by a newer rootstock.
    try:
        ensure_layout_compatible(root)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

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

    # Stamp (or backfill, for pre-marker installs) the layout version.
    write_layout_marker(root)

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
                upgrade=args.upgrade,
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
        upgrade=args.upgrade,
    )
