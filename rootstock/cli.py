"""
Rootstock CLI.

The --root flag specifies the rootstock root directory. If not provided,
the ROOTSTOCK_ROOT environment variable is used.

Commands:
    rootstock install <source> [--root <path>] [--models m1,m2] [--force]
        Install from file (validates, registers, builds):
            rootstock install ./mace_env.py --root /vol/rootstock
        Install all environments from a directory:
            rootstock install ./environments/ --root /vol/rootstock
        Rebuild existing environment by name:
            rootstock install mace_env --root /vol/rootstock --force

    rootstock status [--root <path>]
    rootstock list [--root <path>]
    rootstock serve <model> [--root <path>] --socket <path> --checkpoint <name> [--device <dev>]
    rootstock resolve --cluster <name> [--json]
"""

import argparse
import json as json_module
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .client import RootstockClient
from .clusters import get_cluster_for_root
from .config import (
    DEFAULT_CONFIG_FILE,
    load_config,
    save_config,
)
from .manifest import (
    EnvironmentInfo,
    Manifest,
    compute_source_hash,
    create_manifest,
    get_installed_versions,
    load_manifest,
    now_iso,
    save_manifest,
)
from .pep723 import get_dependencies, get_requires_python

# Environment variable for default root directory
ROOTSTOCK_ROOT_ENV = "ROOTSTOCK_ROOT"


def prompt_with_default(prompt: str, default: str | None = None) -> str | None:
    """Prompt for input with an optional default value."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    value = input(full_prompt).strip()
    if not value and default:
        return default
    return value if value else None


def prompt_secret(prompt: str, existing: str | None = None) -> str | None:
    """Prompt for a secret value without displaying it."""
    if existing:
        # Show that a value exists but don't reveal it
        full_prompt = f"{prompt} [configured]: "
    else:
        full_prompt = f"{prompt}: "

    value = input(full_prompt).strip()
    if not value and existing:
        return existing
    return value if value else None


def cmd_init(args) -> int:
    """
    Interactive initialization of rootstock configuration.

    Prompts user for:
    - Root directory
    - Maintainer name and email
    - API credentials (optional)

    Creates the directory structure and saves config.
    """
    from .clusters import CLUSTER_REGISTRY

    print("Welcome to Rootstock!")
    print("This will help you set up your configuration.\n")

    config = load_config()

    # Prompt for root directory
    print("Root directory is where environments and caches are stored.")
    print(f"Known clusters: {', '.join(CLUSTER_REGISTRY.keys())}")
    print("You can enter a cluster name or a custom path.\n")

    root_default = config.root or os.environ.get(ROOTSTOCK_ROOT_ENV)
    root_input = prompt_with_default("Root directory", root_default)

    if not root_input:
        print("Error: Root directory is required.", file=sys.stderr)
        return 1

    # Check if input is a cluster name
    if root_input in CLUSTER_REGISTRY:
        cluster = root_input
        root = Path(CLUSTER_REGISTRY[root_input])
        print(f"  -> Using cluster '{cluster}' root: {root}")
    else:
        root = Path(root_input).expanduser().resolve()
        cluster = get_cluster_for_root(root)
        if cluster:
            print(f"  -> Detected cluster: {cluster}")

    config.root = str(root)

    print()

    # Prompt for maintainer info
    print("Maintainer information (shown in manifests):")
    config.name = prompt_with_default("  Name", config.name)
    config.email = prompt_with_default("  Email", config.email)

    print()

    # Prompt for API credentials (optional)
    print("API credentials for pushing manifests (optional, press Enter to skip):")
    api_key = prompt_secret("  API Key", config.api_key)
    if api_key:
        config.api_key = api_key
        config.api_secret = prompt_secret("  API Secret", config.api_secret)
        config.api_url = prompt_with_default("  API URL", config.api_url)

    print()

    # Save configuration
    save_config(config)
    print(f"Configuration saved to {DEFAULT_CONFIG_FILE}")

    # Create directory structure
    if not args.skip_dirs:
        print("\nCreating directory structure...")
        dirs_to_create = [
            root / "environments",
            root / "envs",
            root / "cache",
            root / "home",
            root / ".python",
        ]

        for dir_path in dirs_to_create:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"  Created: {dir_path}")
                except PermissionError:
                    print(f"  Skipped (no permission): {dir_path}")
            else:
                print(f"  Exists:  {dir_path}")

    # Initialize manifest if we have a cluster
    if cluster and not args.skip_manifest:
        print("\nInitializing manifest...")
        manifest = create_manifest(root, cluster, config)
        save_manifest(manifest, root)
        print(f"  Created: {root}/manifest.json")

        # Push if configured
        if config.is_push_enabled():
            from .client import RootstockClient

            client = RootstockClient(config)
            success, message = client.push_manifest(manifest)
            if success:
                print(f"  Pushed manifest: {message}")
            else:
                print(f"  Warning: Failed to push: {message}", file=sys.stderr)

    print("\nSetup complete!")
    print("\nNext steps:")
    print("  1. Install environments: rootstock install <env_file.py>")
    print("  2. Check status: rootstock status")

    return 0


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
    models: str | None,
    verbose: bool,
) -> int:
    """
    Install a single environment from a file path or environment name.

    Returns 0 on success, 1 on failure.
    """
    from .environment import get_model_cache_env
    from .pep723 import parse_pep723_metadata, validate_environment_file

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

        # Check if already registered
        if env_source.exists() and not force:
            print(
                f"Error: Environment '{env_name}' already registered at {env_source}",
                file=sys.stderr,
            )
            print("Use --force to update and rebuild", file=sys.stderr)
            return 1

        # Create environments directory and copy file
        env_dir = root / "environments"
        env_dir.mkdir(parents=True, exist_ok=True)
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
    # Find rootstock package path
    import rootstock

    rootstock_path = Path(rootstock.__file__).parent.parent

    result = subprocess.run(
        ["uv", "pip", "install", "--python", str(env_python), str(rootstock_path)],
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

    # Pre-download models if requested
    if models:
        model_list = [m.strip() for m in models.split(",")]
        print(f"5. Pre-downloading models: {model_list}")

        cache_env = get_model_cache_env(root)
        env = {**os.environ, **cache_env}

        for model in model_list:
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
                capture_output=not verbose,
                text=True,
            )
            if result.returncode != 0:
                print(f"   Warning: Failed to download {model}", file=sys.stderr)
                if verbose:
                    print(result.stderr, file=sys.stderr)

    print(f"\nBuilt environment: {env_target}")

    # Update manifest (quiet=True to avoid cluttering build output)
    update_and_push_manifest(root, quiet=False)

    return 0


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
    from .environment import check_uv_available

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
                models=args.models,
                verbose=args.verbose,
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
        models=args.models,
        verbose=args.verbose,
    )


def cmd_status(args) -> int:
    """Show status of rootstock installation."""
    from .environment import list_built_environments, list_environments

    root = get_root_or_exit(args)

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

    # Show config file location
    print(f"\nConfig file: {DEFAULT_CONFIG_FILE}")

    return 0


def cmd_list(args) -> int:
    """List registered environments."""
    from .environment import list_built_environments, list_environments

    root = get_root_or_exit(args)

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


def cmd_resolve(args) -> int:
    """Resolve cluster configuration and print as JSON."""
    import json as json_mod

    from .clusters import get_root_for_cluster

    try:
        root = get_root_for_cluster(args.cluster)
    except ValueError:
        print(f"Error: unknown cluster '{args.cluster}'", file=sys.stderr)
        return 1

    result = {
        "root": str(root),
        "cluster": args.cluster,
    }
    if args.json:
        print(json_mod.dumps(result))
    else:
        print(f"Cluster: {args.cluster}")
        print(f"Root:    {root}")
    return 0


def cmd_serve(args) -> int:
    """
    Start a rootstock worker process for an external i-PI server (e.g., LAMMPS).

    The worker connects to the given Unix socket path, loads the specified model,
    and serves energy/forces via the i-PI protocol until the server disconnects.

    Exit codes:
        0: Clean shutdown
        1: Error
    """
    from .environment import EnvironmentManager

    root = get_root_or_exit(args)
    env_name = f"{args.model}_env"
    socket_path = args.socket
    checkpoint = args.checkpoint
    device = args.device

    # Create environment manager and validate environment exists
    env_mgr = EnvironmentManager(root=root)
    try:
        env_mgr.get_env_python(env_name)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Generate wrapper script
    wrapper_path = env_mgr.generate_wrapper(
        env_name=env_name,
        model=checkpoint,
        device=device,
        socket_path=socket_path,
    )

    # Get spawn command and environment
    cmd = env_mgr.get_spawn_command(env_name, wrapper_path)
    env = env_mgr.get_environment_variables()

    print("Starting rootstock worker:")
    print(f"  Model: {args.model} (env: {env_name})")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device: {device}")
    print(f"  Socket: {socket_path}")

    # Spawn worker subprocess
    proc = subprocess.Popen(cmd, env=env)

    # Forward signals to worker
    def forward_signal(signum, frame):
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    # Block until worker exits
    try:
        rc = proc.wait()
    finally:
        env_mgr.cleanup()

    return rc


def update_and_push_manifest(
    root: Path,
    cluster: str | None = None,
    quiet: bool = False,
) -> bool:
    """
    Update manifest with current state and push to backend.

    Called after any state-changing operation.

    Args:
        root: Rootstock root directory
        cluster: Cluster name (optional, will try to detect)
        quiet: Suppress output

    Returns:
        True if push succeeded or was skipped (no API key), False on error
    """
    config = load_config()

    # Load existing manifest first
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
    manifest = _refresh_manifest_environments(manifest, root)

    # Save locally
    save_manifest(manifest, root)

    # Push to backend if configured
    if config.is_push_enabled():
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

    return True  # No API key = skip push (not an error)


def _refresh_manifest_environments(manifest: Manifest, root: Path) -> Manifest:
    """
    Update manifest with current environment state.

    Scans built environments and updates their info in the manifest.
    """
    from . import __version__
    from .environment import list_built_environments

    # Update rootstock version
    manifest.rootstock_version = __version__

    # Get current built environments
    built = list_built_environments(root)

    for env_name, env_path in built:
        # Check if env_source.py exists
        source_file = env_path / "env_source.py"
        if not source_file.exists():
            continue

        # Get source hash and content
        source_hash = compute_source_hash(source_file)
        source_content = source_file.read_text()

        # Get python requires from source
        python_requires = get_requires_python(source_file) or ">=3.10"

        # Get direct dependencies from source
        direct_deps = get_dependencies(source_file)
        # Always track rootstock itself
        if "rootstock" not in [d.lower() for d in direct_deps]:
            direct_deps.append("rootstock")

        # Get installed package versions (filtered to direct dependencies)
        dependencies = get_installed_versions(env_path, only_packages=direct_deps)

        # Get checkpoints (from existing manifest if available)
        existing_env = manifest.environments.get(env_name)
        checkpoints = existing_env.checkpoints if existing_env else []

        manifest.environments[env_name] = EnvironmentInfo(
            status="ready",
            built_at=existing_env.built_at if existing_env else now_iso(),
            source_hash=source_hash,
            source=source_content,
            python_requires=python_requires,
            dependencies=dependencies,
            checkpoints=checkpoints,
        )

    return manifest


# ============================================================================
# Manifest commands
# ============================================================================


def cmd_manifest(args) -> int:
    """Handle manifest subcommands."""
    if args.manifest_action == "show":
        return cmd_manifest_show(args)
    elif args.manifest_action == "push":
        return cmd_manifest_push(args)
    elif args.manifest_action == "init":
        return cmd_manifest_init(args)
    return 0


def cmd_manifest_show(args) -> int:
    """Show current manifest."""
    root = get_root_or_exit(args)
    manifest = load_manifest(root)

    if manifest is None:
        print(f"No manifest found at {root}/manifest.json", file=sys.stderr)
        print("Run 'rootstock manifest init --cluster <name>' to create one.", file=sys.stderr)
        return 1

    if args.json:
        print(json_module.dumps(manifest.to_dict(), indent=2))
    else:
        print(f"Manifest: {root}/manifest.json")
        print(f"  Schema version:    {manifest.schema_version}")
        print(f"  Cluster:           {manifest.cluster}")
        print(f"  Root:              {manifest.root}")
        print(f"  Rootstock version: {manifest.rootstock_version}")
        print(f"  Python version:    {manifest.python_version}")
        print(f"  Last updated:      {manifest.last_updated}")
        print()
        print("  Maintainer:")
        print(f"    Name:  {manifest.maintainer.name}")
        print(f"    Email: {manifest.maintainer.email}")
        print()
        print(f"  Environments ({len(manifest.environments)}):")
        for name, env in manifest.environments.items():
            print(f"    {name}:")
            print(f"      Status:       {env.status}")
            print(f"      Built at:     {env.built_at}")
            print(f"      Source hash:  {env.source_hash[:20]}...")
            print(f"      Dependencies: {len(env.dependencies)} packages")
            if env.checkpoints:
                print(f"      Checkpoints:  {', '.join(env.checkpoints)}")

    return 0


def cmd_manifest_push(args) -> int:
    """Push manifest to backend."""
    root = get_root_or_exit(args)
    config = load_config()

    # Validate config
    valid, error = config.validate()
    if not valid:
        print(f"Error: {error}", file=sys.stderr)
        print(
            "Run 'rootstock config set --api-key <key> --api-secret <secret> "
            "--api-url <url>' to configure.",
            file=sys.stderr,
        )
        return 1

    manifest = load_manifest(root)
    if manifest is None:
        print(f"No manifest found at {root}/manifest.json", file=sys.stderr)
        return 1

    # Validate manifest
    valid, error = manifest.validate()
    if not valid:
        print(f"Error: Invalid manifest: {error}", file=sys.stderr)
        return 1

    client = RootstockClient(config)
    success, message = client.push_manifest(manifest)

    if success:
        print(message)
        return 0
    else:
        print(f"Error: {message}", file=sys.stderr)
        return 1


def cmd_manifest_init(args) -> int:
    """Initialize manifest for a cluster."""
    root = get_root_or_exit(args)
    cluster = args.cluster
    config = load_config()

    # Check if manifest already exists
    existing = load_manifest(root)
    if existing and not args.force:
        print(f"Error: Manifest already exists at {root}/manifest.json", file=sys.stderr)
        print("Use --force to overwrite.", file=sys.stderr)
        return 1

    # Check maintainer info is configured
    if not config.name or not config.email:
        print("Warning: Maintainer info not configured.", file=sys.stderr)
        print("Run 'rootstock config set --name <name> --email <email>' to set.", file=sys.stderr)

    # Create and save manifest
    manifest = create_manifest(root, cluster, config)
    manifest = _refresh_manifest_environments(manifest, root)
    save_manifest(manifest, root)

    print(f"Manifest initialized: {root}/manifest.json")
    print(f"  Cluster: {cluster}")
    print(f"  Environments: {len(manifest.environments)}")

    # Push if configured
    if config.is_push_enabled():
        client = RootstockClient(config)
        success, message = client.push_manifest(manifest)
        if success:
            print(f"Manifest pushed: {message}")
        else:
            print(f"Warning: Failed to push manifest: {message}", file=sys.stderr)
            print("Run 'rootstock manifest push' to retry.", file=sys.stderr)

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="rootstock",
        description="Rootstock MLIP environment manager",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Interactive setup of rootstock configuration",
        description="Guided setup for root directory, maintainer info, and API credentials.",
    )
    init_parser.add_argument(
        "--skip-dirs",
        action="store_true",
        help="Skip creating directory structure",
    )
    init_parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip initializing manifest",
    )
    init_parser.set_defaults(func=cmd_init)

    # install command
    install_parser = subparsers.add_parser(
        "install",
        help="Install environment(s) from file, directory, or rebuild by name",
        description=(
            "Install environment(s) from a file, directory, or rebuild by name. "
            "File: validates, registers, and builds a single environment. "
            "Directory: installs all *.py environment files. "
            "Name: rebuilds an existing registered environment."
        ),
    )
    install_parser.add_argument(
        "source",
        help="File path, directory, or env name (e.g., ./mace_env.py, ./environments/, mace_env)",
    )
    install_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    install_parser.add_argument("--models", help="Comma-separated list of models to pre-download")
    install_parser.add_argument(
        "--force", action="store_true", help="Update registration and/or rebuild if exists"
    )
    install_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    install_parser.set_defaults(func=cmd_install)

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of rootstock installation",
        description="Show environment sources, built environments, and cache sizes.",
    )
    status_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    status_parser.set_defaults(func=cmd_status)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List registered environments",
        description="List all environment files in the shared environments directory.",
    )
    list_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    list_parser.set_defaults(func=cmd_list)

    # resolve command
    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve cluster configuration",
        description="Look up the root directory for a known cluster.",
    )
    resolve_parser.add_argument("--cluster", required=True, help="Cluster name")
    resolve_parser.add_argument("--json", action="store_true", help="Output as JSON")
    resolve_parser.set_defaults(func=cmd_resolve)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a worker for an external i-PI server",
        description="Start a rootstock worker that connects to a Unix socket.",
    )
    serve_parser.add_argument("model", help="Model family (e.g., mace, uma, tensornet)")
    serve_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    serve_parser.add_argument("--socket", required=True, help="Unix socket path to connect to")
    serve_parser.add_argument("--checkpoint", required=True, help="Checkpoint/weights name")
    serve_parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    serve_parser.set_defaults(func=cmd_serve)

    # manifest command
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Manage installation manifest",
        description="Manage the manifest that tracks installation state.",
    )
    manifest_subparsers = manifest_parser.add_subparsers(
        dest="manifest_action",
        required=True,
    )

    # manifest show
    manifest_show_parser = manifest_subparsers.add_parser(
        "show",
        help="Show current manifest",
    )
    manifest_show_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    manifest_show_parser.add_argument("--json", action="store_true", help="Output as JSON")
    manifest_show_parser.set_defaults(func=cmd_manifest)

    # manifest push
    manifest_push_parser = manifest_subparsers.add_parser(
        "push",
        help="Push manifest to backend",
    )
    manifest_push_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    manifest_push_parser.set_defaults(func=cmd_manifest)

    # manifest init
    manifest_init_parser = manifest_subparsers.add_parser(
        "init",
        help="Initialize manifest for a cluster",
    )
    manifest_init_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    manifest_init_parser.add_argument(
        "--cluster",
        required=True,
        help="Cluster name (e.g., della, modal)",
    )
    manifest_init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest",
    )
    manifest_init_parser.set_defaults(func=cmd_manifest)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
