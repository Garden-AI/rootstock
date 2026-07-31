"""
Operations layer: build / add / refresh as callable functions.

Commands in ``rootstock/commands/`` are thin argparse adapters over this
module; external drivers import these functions instead of shelling out to
the CLI and screen-scraping its output.

Conventions:
- Keyword parameters only — no argparse Namespaces.
- Progress is reported through an optional ``progress`` callable receiving
  one line of text per event; nothing is emitted when it is None. Warnings
  that must reach a human even in programmatic use go to stderr.
- Failures raise :class:`OperationError` with a user-presentable message
  (domain errors like ``CheckpointNotFoundError`` pass through).
- Callers own process-wide concerns: umask for shared installs, uv
  availability, and layout-compatibility checks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .client import RootstockClient
from .clusters import get_cluster_for_root
from .config import load_config
from .environment import (
    declares_setup_from_path,
    find_env_for_checkpoint,
    parse_checkpoints_dict,
    parse_custom_checkpoint_ids,
)
from .exceptions import RootstockError
from .layout import resolve_cache_root
from .manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    Manifest,
    built_at_estimate,
    create_manifest,
    get_installed_versions,
    load_manifest,
    manifest_lock,
    now_iso,
    save_manifest,
)
from .pep723 import (
    get_dependencies,
    get_requires_python,
    parse_pep723_metadata,
    validate_environment_file,
)
from .spawn import DOWNLOAD_WRAPPER, spawn_in_env
from .verify import verify_checkpoint

ROOTSTOCK_GITHUB_URL = "https://github.com/Garden-AI/rootstock.git"

Progress = Callable[[str], None]


class OperationError(RootstockError, RuntimeError):
    """An operation failed; the message is presentable to the user as-is."""


@dataclass
class InstallResult:
    """Outcome of a successful :func:`install_environment`."""

    env_name: str
    env_path: Path
    rebuilt: bool  # an existing built env was replaced
    locked: bool  # dependencies were installed from a resolved lockfile


@dataclass
class AddResult:
    """Outcome of a successful :func:`add_checkpoint`."""

    env_name: str
    checkpoint: str
    fetched_at: str
    already_fetched: bool  # download skipped: weights were already cached
    verified_at: str | None  # None when verification was skipped
    verified_device: str | None


@dataclass
class FetchResult:
    """Outcome of a successful :func:`fetch_checkpoint`."""

    env_name: str
    checkpoint: str
    fetched_at: str
    already_fetched: bool  # download skipped: weights were already cached


@dataclass
class VerifyResult:
    """Outcome of a successful :func:`verify_fetched_checkpoint`."""

    env_name: str
    checkpoint: str
    verified_at: str
    verified_device: str


def _say(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


# -----------------------------------------------------------------------------
# setup_kwargs parsing (shared by the add and serve commands)
# -----------------------------------------------------------------------------


def parse_kwarg(spec: str) -> tuple[str, object]:
    """
    Parse a ``key=val`` spec. Value is JSON-decoded first; on parse
    failure it falls back to the raw string. So::

        task=omat       -> ("task", "omat")
        charge=-1       -> ("charge", -1)
        enabled=true    -> ("enabled", True)
        scale=1.5       -> ("scale", 1.5)
    """
    if "=" not in spec:
        raise ValueError(f"--kwarg expects key=value, got {spec!r}")
    key, _, raw = spec.partition("=")
    if not key:
        raise ValueError(f"--kwarg key cannot be empty: {spec!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def parse_setup_kwargs(kwarg_specs: list[str] | None) -> dict[str, object]:
    """Parse repeated ``key=val`` specs into a setup_kwargs dict."""
    out: dict[str, object] = {}
    for spec in kwarg_specs or []:
        key, value = parse_kwarg(spec)
        out[key] = value
    return out


# -----------------------------------------------------------------------------
# Manifest refresh + push
# -----------------------------------------------------------------------------


def refresh_manifest_environments(
    manifest: Manifest, root: Path, built_env: str | None = None
) -> Manifest:
    """
    Update manifest with current environment state.

    Reads the installation through the InstallState reader: the filesystem
    decides which envs the manifest records, and records for envs no longer
    on disk are dropped.

    built_at semantics: `built_env` (the env the calling command just built)
    is stamped now; envs already in the manifest keep their recorded time; an
    env the manifest has never seen gets the env directory's mtime as a
    best-effort estimate — never `now`, which would fake freshness into the
    `verified_at > built_at` staleness comparison.
    """
    from . import __version__
    from .install_state import read_install_state

    # Update rootstock version
    manifest.rootstock_version = __version__

    state = read_install_state(root, manifest=manifest)

    for env_name, env in state.envs.items():
        if env.source_file is None:
            # Built but missing env_source.py — nothing derivable to record;
            # keep whatever record already exists.
            continue

        source_content = env.source_file.read_text()

        # Get python requires from source
        python_requires = get_requires_python(env.source_file) or ">=3.11"

        # Get direct dependencies from source
        direct_deps = get_dependencies(env.source_file)
        # Always track rootstock itself
        if "rootstock" not in [d.lower() for d in direct_deps]:
            direct_deps.append("rootstock")

        # Get installed package versions (filtered to direct dependencies)
        dependencies = get_installed_versions(env.path, only_packages=direct_deps)

        # Get checkpoints (from existing manifest if available)
        checkpoints = env.record.checkpoints if env.record else {}

        if env_name == built_env:
            built_at = now_iso()
        elif env.record:
            built_at = env.record.built_at
        else:
            built_at = built_at_estimate(env.path)

        manifest.environments[env_name] = EnvironmentInfo(
            built_at=built_at,
            source_hash=env.source_hash,
            source=source_content,
            python_requires=python_requires,
            dependencies=dependencies,
            checkpoints=checkpoints,
            lock_hash=env.lock_hash,
        )

    # The filesystem is the truth for what's installed: a record whose env
    # is gone from disk describes nothing and must not reach the push payload.
    for env_name in sorted(state.manifest_only_envs):
        print(
            f"Note: dropping manifest record for '{env_name}' — env no longer on disk",
            file=sys.stderr,
        )
        del manifest.environments[env_name]

    return manifest


def update_and_push_manifest(
    root: Path,
    cluster: str | None = None,
    quiet: bool = False,
    push: bool = True,
    built_env: str | None = None,
) -> bool:
    """
    Update manifest with current state and optionally push to backend.

    Called after any state-changing operation.

    Args:
        root: Rootstock root directory
        cluster: Cluster name (optional, will try to detect)
        quiet: Suppress output
        push: Whether to push to backend (default True)
        built_env: Env name that was (re)built by the calling command, if any;
            its built_at is stamped to now

    Returns:
        True if push succeeded or was skipped (no API key), False on error
    """
    config = load_config()

    # The whole load → refresh → save cycle runs under the manifest lock so a
    # concurrent writer (nightly smoke-test vs. a co-maintainer's add) can't
    # be silently overwritten. The push happens after release — no reason to
    # hold the lock across a network call.
    with manifest_lock(root):
        manifest = load_manifest(root)

        # Determine cluster: provided > existing manifest > detect from path
        if cluster is None:
            if manifest is not None:
                cluster = manifest.cluster
            else:
                cluster = get_cluster_for_root(root)

        if cluster is None:
            if not quiet:
                print(
                    "Warning: Cannot update manifest - cluster not specified and "
                    "root doesn't match any known cluster. "
                    "Run 'rootstock manifest init --cluster <name>' first.",
                    file=sys.stderr,
                )
            return False

        # Create manifest if it doesn't exist
        if manifest is None:
            manifest = create_manifest(root, cluster, config)

        # Refresh environment info from current state
        manifest = refresh_manifest_environments(manifest, root, built_env=built_env)

        # Save locally
        save_manifest(manifest, root)

    # Skip push if explicitly disabled
    if not push:
        if not quiet:
            print("Manifest saved locally (push disabled via --no-push).")
        return True

    # Push to backend only if user is maintainer and API is configured
    if config.is_maintainer and config.is_push_enabled():
        client = RootstockClient(config)
        success, message = client.push_manifest(manifest)
        if not quiet:
            if success:
                print(f"Manifest pushed: {message}")
            else:
                print(
                    f"Warning: Failed to push manifest: {message}",
                    file=sys.stderr,
                )
                print(
                    "Manifest saved locally. Run 'rootstock manifest push' to retry.",
                    file=sys.stderr,
                )
        return success

    # Not the configured maintainer (or no API credentials): saving locally
    # is the normal, expected outcome — nothing to warn about.
    return True


# -----------------------------------------------------------------------------
# Environment install
# -----------------------------------------------------------------------------


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


def _vendor_rootstock_wheel(root: Path) -> Path | None:
    """Archive this release's wheel in {root}/wheels/ and return its path.

    Rebuilding an env reinstalls the pinned rootstock, which normally depends
    on PyPI still serving that exact release, un-yanked, years later. Keeping
    the wheel inside the install root removes that dependency: rebuilds
    install rootstock from the vendored file, and the wheel doubles as a
    bootstrap source if the release ever disappears upstream.

    Returns None (with a warning) when vendoring isn't possible: dev builds
    have no published wheel (they pin a git sha instead), and a download or
    checksum failure must not fail the install — it just falls back to the
    index. The download is verified against PyPI's published sha256.
    """
    import hashlib
    import urllib.error
    import urllib.request

    from rootstock import __version__

    if "dev" in __version__:
        return None

    wheels_dir = root / "wheels"
    existing = sorted(wheels_dir.glob(f"rootstock-{__version__}-*.whl"))
    if existing:
        return existing[0]

    try:
        url = f"https://pypi.org/pypi/rootstock/{__version__}/json"
        with urllib.request.urlopen(url, timeout=30) as resp:
            release = json.load(resp)
        wheel_meta = next(
            entry for entry in release["urls"] if entry["packagetype"] == "bdist_wheel"
        )
        with urllib.request.urlopen(wheel_meta["url"], timeout=60) as resp:
            wheel_bytes = resp.read()

        digest = hashlib.sha256(wheel_bytes).hexdigest()
        if digest != wheel_meta["digests"]["sha256"]:
            raise ValueError(
                f"sha256 mismatch for {wheel_meta['filename']}: "
                f"got {digest}, PyPI says {wheel_meta['digests']['sha256']}"
            )

        wheels_dir.mkdir(parents=True, exist_ok=True)
        wheel_path = wheels_dir / wheel_meta["filename"]
        tmp_path = wheel_path.with_suffix(".whl.partial")
        tmp_path.write_bytes(wheel_bytes)
        tmp_path.rename(wheel_path)
        return wheel_path
    except (urllib.error.URLError, OSError, ValueError, KeyError, StopIteration) as exc:
        print(
            f"  Warning: could not vendor the rootstock wheel into {wheels_dir} "
            f"({exc}); this build will install rootstock from the index instead.",
            file=sys.stderr,
        )
        return None


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


def _publish_python_interpreter(
    tmp_python_dir: Path,
    python_install_dir: Path,
    progress: Progress | None,
) -> None:
    """Move a freshly downloaded interpreter into the shared ``.python/`` dir.

    Parallel builds race here (uv downloads each build's interpreter to a
    private tempdir; publication into the shared dir is ours to make safe):
    each item is copied to a unique staging name *inside* ``.python/`` (same
    filesystem) and atomically renamed into place, so a concurrent reader —
    another build's ``uv venv`` — can never see a half-copied interpreter
    tree. Losing the rename race to another build is fine: its copy is as
    good as ours, so the staging copy is simply discarded.

    Once the destination exists, leftover staging entries for that item
    (from builds that crashed mid-copy) are swept. Sweeping a *live* loser's
    staging is harmless: its copy or rename fails, it sees the destination
    exists, and it carries on.
    """
    if not tmp_python_dir.exists():
        return

    for item in tmp_python_dir.iterdir():
        dest = python_install_dir / item.name
        if not dest.exists():
            _say(progress, f"  Copying Python to {dest}")
            # pid alone is not unique enough: parallel builds may be threads
            # of one process (a batch driver), so salt with the thread id.
            staging = python_install_dir / (
                f"{item.name}.installing.{os.getpid()}.{threading.get_ident()}"
            )
            try:
                if item.is_dir():
                    shutil.copytree(item, staging)
                else:
                    shutil.copy2(item, staging)
                staging.rename(dest)
            except OSError:
                # Tolerable only when another build actually won the race.
                if not dest.exists():
                    raise
        # dest exists now — ours or another build's. Anything still staged
        # for this item is dead weight (a crashed build's leftover, or our
        # own copy after losing the race); reclaim the space.
        for stale in python_install_dir.glob(f"{item.name}.installing.*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)


def install_environment(
    root: Path,
    source: str | Path,
    *,
    force: bool = False,
    upgrade: bool = False,
    verbose: bool = False,
    push: bool = True,
    progress: Progress | None = None,
) -> InstallResult:
    """
    Install (build) a single environment from a file path or registered name.

    A file path is validated, registered into ``{root}/environments/``, and
    built; a bare name rebuilds an already-registered environment. The build
    happens in ``{root}/.build`` and is atomically swapped into
    ``{root}/envs/<name>`` — a failed rebuild leaves the live env untouched.

    Raises OperationError on failure. The caller owns process-wide setup
    (umask for shared installs, uv availability, layout compatibility).
    """
    root = Path(root)
    source_path = Path(source)

    # Determine mode: file path or environment name
    if source_path.is_file():
        # FILE MODE: validate → copy → build
        env_name = source_path.stem
        env_source = root / "environments" / f"{env_name}.py"

        _say(progress, f"Validating {source_path}...")
        is_valid, error = validate_environment_file(source_path)
        if not is_valid:
            raise OperationError(str(error))

        try:
            parse_checkpoints_dict(source_path)
        except ValueError as exc:
            raise OperationError(str(exc)) from exc

        # '<family>:custom' entries and the setup_from_path hook only work
        # together: an entry without the hook would resolve (and be listed)
        # but always fail to load.
        custom_ids = parse_custom_checkpoint_ids(source_path)
        declares_hook = declares_setup_from_path(source_path)
        if custom_ids and not declares_hook:
            raise OperationError(
                f"{source_path} declares {', '.join(custom_ids)} in "
                f"CHECKPOINTS but no setup_from_path() hook — add the hook "
                f"or drop the entries."
            )
        if declares_hook and not custom_ids:
            _say(
                progress,
                f"note: {env_name} declares setup_from_path() but no "
                f"'<family>:custom' CHECKPOINTS entry — user-supplied "
                f"weights stay unavailable until one is declared.",
            )

        # If the source file is already the registered file (the natural flow
        # on a shared install: drop file into <root>/environments/ then run
        # install), skip the copy and the "already registered" guard.
        already_at_canonical = env_source.exists() and source_path.resolve() == env_source.resolve()

        if env_source.exists() and not force and not already_at_canonical:
            raise OperationError(
                f"Environment '{env_name}' already registered at {env_source}\n"
                "Use --force to update and rebuild"
            )

        # Create environments directory and copy file (unless already there)
        env_dir = root / "environments"
        env_dir.mkdir(parents=True, exist_ok=True)
        if not already_at_canonical:
            shutil.copy2(source_path, env_source)
            _say(progress, f"Registered: {source_path} -> {env_source}")
            # A lockfile shipped alongside the source (e.g. tracked in a git
            # repo next to the env file) is authoritative — carry it along so
            # the build resolves from it instead of from scratch.
            source_lock = _lockfile_for(source_path)
            if source_lock.exists():
                shutil.copy2(source_lock, _lockfile_for(env_source))
                _say(progress, f"Registered lockfile: {source_lock} -> {_lockfile_for(env_source)}")

    else:
        # NAME MODE: use existing registered environment
        env_name = str(source)
        env_source = root / "environments" / f"{env_name}.py"

        if not env_source.exists():
            message = f"Environment not found: {env_name}"
            available = (
                list((root / "environments").glob("*.py"))
                if (root / "environments").exists()
                else []
            )
            if available:
                message += f"\nAvailable: {[p.stem for p in available]}"
            raise OperationError(message)

    env_target = root / "envs" / env_name

    # Check if venv already exists. A --force rebuild does NOT delete the live
    # env here — the new env is built in {root}/.build and swapped in at the
    # end, so worker spawns on a shared install keep working for the whole
    # (slow) build, and a failed rebuild leaves the old env untouched.
    rebuilt = env_target.exists()
    if rebuilt and not force:
        raise OperationError(f"Environment already built: {env_target}\nUse --force to rebuild")

    build_root = root / ".build"
    build_root.mkdir(parents=True, exist_ok=True)
    # Clear leftovers from earlier crashed/killed builds of this env.
    for stale in build_root.glob(f"{env_name}.*"):
        shutil.rmtree(stale, ignore_errors=True)
    build_dir = build_root / f"{env_name}.{os.getpid()}"

    _say(progress, f"Building environment: {env_name}")
    _say(progress, f"  Source: {env_source}")
    _say(progress, f"  Target: {env_target}")
    if rebuilt:
        _say(progress, f"  (building in {build_dir}; the live env is swapped out only when done)")

    try:
        locked = _build_and_swap(
            root=root,
            env_name=env_name,
            env_source=env_source,
            env_target=env_target,
            build_dir=build_dir,
            verbose=verbose,
            push=push,
            upgrade=upgrade,
            progress=progress,
        )
    finally:
        # Gone already on success (renamed into place); left behind on any
        # failure path or exception.
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)

    return InstallResult(env_name=env_name, env_path=env_target, rebuilt=rebuilt, locked=locked)


def _build_and_swap(
    root: Path,
    env_name: str,
    env_source: Path,
    env_target: Path,
    build_dir: Path,
    verbose: bool,
    push: bool,
    upgrade: bool,
    progress: Progress | None,
) -> bool:
    """Build the venv into build_dir, then atomically swap it into env_target.

    Returns whether dependencies were installed from a resolved lockfile.
    """
    # Parse PEP 723 metadata
    content = env_source.read_text()
    metadata = parse_pep723_metadata(content)
    if metadata is None:
        raise OperationError(f"No PEP 723 metadata in {env_source}")

    dependencies = metadata.get("dependencies", [])
    requires_python = metadata.get("requires-python", ">=3.11")

    # Extract minimum version properly
    try:
        python_version = extract_minimum_python_version(requires_python)
    except ValueError as exc:
        raise OperationError(str(exc)) from exc

    _say(progress, f"  Python: {requires_python} -> {python_version}")
    _say(progress, f"  Dependencies: {dependencies}")

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
    _say(progress, "1. Creating virtual environment...")
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
            raise OperationError(f"Error downloading Python: {result.stderr}")

        # Publish the downloaded Python into the shared root (if not already
        # there) — staged-copy + atomic rename, safe against parallel builds.
        _publish_python_interpreter(tmp_python_dir, python_install_dir, progress)

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
        raise OperationError(f"Error creating venv: {result.stderr}")

    env_python = build_dir / "bin" / "python"

    # Resolve the dependency lockfile. A build is only as reproducible as its
    # resolution: without a lockfile, a rebuild months later re-resolves the
    # env file's version ranges and produces a different env. `uv lock` keeps
    # the pins in an existing lockfile (re-resolving only what a source edit
    # forces), creates the lockfile on first build, and re-resolves everything
    # only with --upgrade.
    lock_path = _lockfile_for(env_source)
    locked = False
    if dependencies:
        _say(progress, "2. Resolving dependency lockfile...")
        if upgrade:
            _say(progress, f"  --upgrade: re-resolving all pins in {lock_path.name}")
        elif lock_path.exists():
            _say(progress, f"  Honoring existing lockfile: {lock_path}")
        else:
            _say(progress, f"  No lockfile yet; resolving and writing {lock_path}")

        lock_cmd = ["uv", "lock", "--script", str(env_source)]
        if upgrade:
            lock_cmd.append("--upgrade")
        result = subprocess.run(
            lock_cmd,
            capture_output=not verbose,
            text=True,
            env=uv_env,
        )
        locked = result.returncode == 0
        if not locked:
            # `uv lock` resolves for every platform at once; envs that pull
            # prebuilt wheels from a platform-specific index (e.g. the PyG
            # find-links pages, which ship no macOS wheels) cannot be locked
            # at all. That alone must not fail the build — sync gets to try
            # below with its own resolution, so a genuinely broken dependency
            # still fails there, loudly, instead of being masked here.
            print(
                "  Warning: could not resolve a lockfile for this env "
                "(expected for envs whose find-links index lacks wheels for "
                "some platform; otherwise the dependency error will surface "
                "in the next step). Falling back to uv sync's own resolution.",
                file=sys.stderr,
            )
            if not verbose and result.stderr:
                print(f"  uv lock said: {result.stderr.strip().splitlines()[-1]}", file=sys.stderr)
    else:
        _say(progress, "2. Resolving dependency lockfile... (no dependencies, skipped)")

    # Install dependencies with `uv sync --script` so the env source's full
    # PEP 723 uv config is honored — not just `dependencies`, but
    # `[tool.uv.sources]`, `[[tool.uv.index]]`, and `[tool.uv]` find-links.
    # The `uv pip` interface silently ignores sources/index pins. `--active` +
    # VIRTUAL_ENV targets the venv we just created. When the lock step
    # succeeded, `--frozen` installs exactly its pins — sync must never
    # re-resolve on its own. When it failed, sync resolves for itself (and a
    # valid pre-existing lockfile is still validated and used); --frozen here
    # would silently install stale pins for a source whose real resolution
    # just failed.
    _say(progress, "3. Installing dependencies...")

    if dependencies:
        sync_cmd = ["uv", "sync", "--script", str(env_source), "--active"]
        if locked:
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
            raise OperationError(
                f"Error installing dependencies: {result.stderr if not verbose else ''}"
            )

    # Install rootstock. Prefer the wheel vendored into {root}/wheels/ (this
    # release's own artifact, archived so rebuilds don't depend on PyPI still
    # serving it); fall back to the pinned index/git spec.
    _say(progress, "4. Installing rootstock...")
    vendored_wheel = _vendor_rootstock_wheel(root)
    if vendored_wheel is not None:
        rootstock_spec = str(vendored_wheel)
        _say(progress, f"  Installing vendored wheel: {vendored_wheel}")
    else:
        rootstock_spec = _rootstock_install_spec()
        _say(progress, f"  Installing: {rootstock_spec}")

    result = subprocess.run(
        ["uv", "pip", "install", "--python", str(env_python), rootstock_spec],
        capture_output=not verbose,
        text=True,
        env=uv_env,
    )
    if result.returncode != 0:
        raise OperationError(f"Error installing rootstock: {result.stderr if not verbose else ''}")

    # Copy environment source file (and its lockfile, so the built env records
    # exactly what it was resolved from)
    _say(progress, "5. Copying environment source...")
    shutil.copy(env_source, build_dir / "env_source.py")
    if dependencies and lock_path.exists():
        shutil.copy(lock_path, build_dir / "env_source.py.lock")

    # Swap the finished build into place. Two renames (same filesystem, so
    # each is atomic) shrink the unavailable window from the whole build to
    # microseconds; a failure before this point never touched the live env.
    _say(progress, "6. Swapping new environment into place...")
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

    _say(progress, "7. Pre-compiling bytecode...")
    _precompile_environment(env_python, env_target)

    _say(progress, f"\nBuilt environment: {env_target}")

    # Update manifest. built_env stamps this env's built_at to now — the one
    # moment the true build time is known.
    update_and_push_manifest(root, quiet=progress is None, push=push, built_env=env_name)

    return locked


# -----------------------------------------------------------------------------
# Checkpoint add (download + verify)
# -----------------------------------------------------------------------------


def _run_download(
    root: Path,
    env_name: str,
    checkpoint: str,
    setup_kwargs: dict,
    cache_root: Path | None = None,
) -> tuple[bool, str | None]:
    """Run ``setup(checkpoint, "cpu", **setup_kwargs)`` to trigger the cache-aware
    download path. Returns ``(ok, error)``."""
    env_dir = root / "envs" / env_name
    if not (env_dir / "bin" / "python").exists():
        return False, f"environment not built at {env_dir}"

    with spawn_in_env(
        root,
        env_name,
        DOWNLOAD_WRAPPER,
        {"checkpoint": checkpoint, "device": "cpu", "setup_kwargs": setup_kwargs},
        cache_root=cache_root,
    ) as spec:
        result = subprocess.run(
            spec.cmd,
            env=spec.env,
            cwd=spec.cwd,
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        err = result.stderr.strip().splitlines()
        # Last line is usually the most informative (the actual exception message).
        msg = err[-1] if err else f"setup() exited with code {result.returncode}"
        return False, msg
    return True, None


@contextmanager
def _manifest_transaction(root: Path, cluster_hint: str | None = None):
    """Lock → load-fresh (or create) → yield → save-once → unlock.

    The single sanctioned way to mutate the manifest. Long-running work
    (downloads, verification subprocesses) must happen *outside* the
    transaction; the body should only apply its results. On an exception in
    the body nothing is saved.
    """
    with manifest_lock(root):
        manifest = load_manifest(root)
        if manifest is None:
            config = load_config()
            manifest = create_manifest(root, cluster_hint or "unknown", config)
        yield manifest
        save_manifest(manifest, root)


def _ensure_manifest_entry(
    manifest: Manifest,
    root: Path,
    env_name: str,
    checkpoint: str,
) -> tuple[EnvironmentInfo, CheckpointInfo]:
    """Return the env+checkpoint records, refreshing from disk if the env is
    missing from the manifest. Mutates ``manifest`` in place — call inside a
    transaction."""
    env = manifest.environments.get(env_name)
    if env is None:
        # The manifest hasn't been refreshed since this env was built. Refresh
        # from disk rather than synthesizing a placeholder — a fabricated
        # built_at=now would fake freshness into the verified_at > built_at
        # staleness comparison, and an empty source_hash would record nothing.
        env_dir = root / "envs" / env_name
        if not (env_dir / "bin" / "python").exists():
            raise OperationError(
                f"environment '{env_name}' is not built at {env_dir}.\n"
                f"Run: rootstock install <path-to-{env_name}.py> --root {root}"
            )
        manifest = refresh_manifest_environments(manifest, root)
        env = manifest.environments.get(env_name)
        if env is None:
            raise OperationError(
                f"environment '{env_name}' is built at {env_dir} but has no "
                f"env_source.py — rebuild it: rootstock install {env_name} "
                f"--root {root} --force"
            )

    if checkpoint not in env.checkpoints:
        env.checkpoints[checkpoint] = CheckpointInfo()
    return env, env.checkpoints[checkpoint]


def _resolve_built_env(root: Path, checkpoint: str) -> str:
    """Resolve the hosting env for a checkpoint id, failing fast when it
    isn't built (checked again inside each transaction by
    :func:`_ensure_manifest_entry`, whose refresh needs it too)."""
    env_name, _ = find_env_for_checkpoint(root, checkpoint)
    env_dir = root / "envs" / env_name
    if not (env_dir / "bin" / "python").exists():
        raise OperationError(
            f"environment '{env_name}' is not built at {env_dir}.\n"
            f"Run: rootstock install <path-to-{env_name}.py> --root {root}"
        )
    return env_name


def fetch_checkpoint(
    root: Path,
    checkpoint: str,
    *,
    setup_kwargs: dict | None = None,
    cache_root: Path | None = None,
    refresh: bool = True,
    push: bool = True,
    force: bool = False,
    progress: Progress | None = None,
) -> FetchResult:
    """
    Idempotent CPU-side download of a checkpoint by canonical id.

    The download half of :func:`add_checkpoint`, usable on its own: it needs
    no GPU (so it can run on a login node), and downloads parallelize freely
    — HF hub uses its own file locks, and distinct checkpoints touch distinct
    cache files — where verification must stay bounded by GPU memory.

    Records ``fetched_at``/``last_error`` in the manifest. ``refresh=False``
    skips the trailing full manifest refresh (+ push): a batch driver running
    many fetches refreshes once at the end instead of once per checkpoint.

    ``force=True`` re-runs the download even when the manifest records the
    checkpoint as fetched — the repair path for cache files that have gone
    missing behind the manifest's back (cleaned, or an interrupted download
    that stamped anyway). The underlying download is cache-aware, so a
    forced fetch of an intact checkpoint costs a cache hit, not a transfer.

    Raises CheckpointNotFoundError when no installed env declares the id,
    and OperationError when the download fails (also recorded in the
    manifest's ``last_error``).
    """
    root = Path(root)
    setup_kwargs = setup_kwargs or {}
    if cache_root is None:
        cache_root = resolve_cache_root(root)

    env_name = _resolve_built_env(root, checkpoint)

    # Unlocked peek for idempotence; every write below loads fresh inside
    # its own transaction, so a racing writer costs at most a redundant
    # (idempotent) download, never a lost update.
    peek = load_manifest(root)
    peek_env = peek.environments.get(env_name) if peek else None
    peek_ckpt = peek_env.checkpoints.get(checkpoint) if peek_env else None
    already_fetched = not force and peek_ckpt is not None and peek_ckpt.fetched_at is not None
    fetched_at = peek_ckpt.fetched_at if peek_ckpt else None

    # ---- Download (runs outside the lock) -------------------------------
    if not already_fetched:
        _say(progress, f"Downloading {env_name}/{checkpoint} on CPU...")
        ok, err = _run_download(root, env_name, checkpoint, setup_kwargs, cache_root=cache_root)
        with _manifest_transaction(root) as manifest:
            _, ckpt = _ensure_manifest_entry(manifest, root, env_name, checkpoint)
            if ok:
                ckpt.fetched_at = now_iso()
                ckpt.last_error = None
                fetched_at = ckpt.fetched_at
            else:
                ckpt.last_error = f"download: {err}"
        if not ok:
            raise OperationError(f"download failed: {err}")
        _say(progress, f"  fetched_at = {fetched_at}")
    else:
        _say(progress, f"{env_name}/{checkpoint} already fetched at {fetched_at}")

    if refresh:
        # Refresh + push (takes the lock for its own load -> refresh -> save)
        update_and_push_manifest(root, quiet=True, push=push)

    # Every path here either set fetched_at (fresh download) or found it
    # already recorded (already_fetched); a failed download raised above.
    assert fetched_at is not None
    return FetchResult(
        env_name=env_name,
        checkpoint=checkpoint,
        fetched_at=fetched_at,
        already_fetched=already_fetched,
    )


def verify_fetched_checkpoint(
    root: Path,
    checkpoint: str,
    *,
    device: str = "cuda",
    setup_kwargs: dict | None = None,
    cache_root: Path | None = None,
    refresh: bool = True,
    push: bool = True,
    progress: Progress | None = None,
) -> VerifyResult:
    """
    Verify a checkpoint on ``device`` and record the outcome in the manifest.

    The verify half of :func:`add_checkpoint`, usable on its own — typically
    after :func:`fetch_checkpoint` (a prior fetch isn't a hard precondition;
    verification loads through the same cache-aware path and would download
    missing weights itself). Each verification loads the full model onto
    ``device``, so a batch driver must bound its concurrency, unlike the
    freely-parallel fetch.

    Success stamps ``verified_at``/``verified_device`` and clears
    ``last_error``; failure clears the verified stamps and records the error.
    ``refresh=False`` skips the trailing full manifest refresh (+ push), for
    batch drivers that refresh once at the end.

    Raises CheckpointNotFoundError when no installed env declares the id,
    and OperationError when verification fails.
    """
    root = Path(root)
    setup_kwargs = setup_kwargs or {}
    if cache_root is None:
        cache_root = resolve_cache_root(root)

    env_name = _resolve_built_env(root, checkpoint)

    # ---- Verify (runs outside the lock) ---------------------------------
    _say(progress, f"Verifying {env_name}/{checkpoint} on {device}...")
    ok, err = verify_checkpoint(
        root, env_name, checkpoint, device, setup_kwargs, cache_root=cache_root
    )
    with _manifest_transaction(root) as manifest:
        _, ckpt = _ensure_manifest_entry(manifest, root, env_name, checkpoint)
        if ok:
            ckpt.verified_at = now_iso()
            ckpt.verified_device = device
            ckpt.last_error = None
            verified_at = ckpt.verified_at
        else:
            ckpt.verified_at = None
            ckpt.verified_device = None
            ckpt.last_error = f"verify: {err}"
    if not ok:
        raise OperationError(f"verify failed: {err}")
    _say(progress, f"  verified_at = {verified_at} ({device})")

    if refresh:
        # Refresh + push (takes the lock for its own load -> refresh -> save)
        update_and_push_manifest(root, quiet=True, push=push)

    return VerifyResult(
        env_name=env_name,
        checkpoint=checkpoint,
        verified_at=verified_at,
        verified_device=device,
    )


def add_checkpoint(
    root: Path,
    checkpoint: str,
    *,
    device: str = "cuda",
    verify: bool = True,
    push: bool = True,
    force: bool = False,
    setup_kwargs: dict | None = None,
    cache_root: Path | None = None,
    progress: Progress | None = None,
) -> AddResult:
    """
    Idempotent download-or-verify for a checkpoint by canonical id.

    The composition of :func:`fetch_checkpoint` and
    :func:`verify_fetched_checkpoint` (batch drivers call those directly to
    parallelize the two phases independently), with a single manifest
    refresh at the end. The hosting env is resolved by matching the id
    against each installed env's CHECKPOINTS table. Downloads happen on CPU
    (the cache-aware path); verification runs on ``device`` unless
    ``verify`` is False. ``force`` re-runs the download past the manifest's
    fetched stamp (see :func:`fetch_checkpoint`).

    Raises CheckpointNotFoundError when no installed env declares the id,
    and OperationError for download/verify failures (the failure is also
    recorded in the manifest's ``last_error`` for the checkpoint; a failure
    skips the trailing refresh, exactly as before the split).
    """
    root = Path(root)
    setup_kwargs = setup_kwargs or {}
    if cache_root is None:
        cache_root = resolve_cache_root(root)

    fetched = fetch_checkpoint(
        root,
        checkpoint,
        setup_kwargs=setup_kwargs,
        cache_root=cache_root,
        refresh=False,
        force=force,
        progress=progress,
    )

    verified: VerifyResult | None = None
    if not verify:
        _say(progress, "(skipping verify per --no-verify)")
    else:
        verified = verify_fetched_checkpoint(
            root,
            checkpoint,
            device=device,
            setup_kwargs=setup_kwargs,
            cache_root=cache_root,
            refresh=False,
            progress=progress,
        )

    # Refresh + push (takes the lock for its own load -> refresh -> save)
    update_and_push_manifest(root, quiet=True, push=push)

    return AddResult(
        env_name=fetched.env_name,
        checkpoint=checkpoint,
        fetched_at=fetched.fetched_at,
        already_fetched=fetched.already_fetched,
        verified_at=verified.verified_at if verified else None,
        verified_device=verified.verified_device if verified else None,
    )
