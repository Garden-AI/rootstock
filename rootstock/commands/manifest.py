"""Manifest management commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..client import RootstockClient
from ..clusters import get_cluster_for_root
from ..config import load_config
from ..manifest import (
    EnvironmentInfo,
    Manifest,
    built_at_estimate,
    compute_source_hash,
    create_manifest,
    get_installed_versions,
    load_manifest,
    now_iso,
    save_manifest,
)
from ..pep723 import get_dependencies, get_requires_python
from .common import get_root_or_exit


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
    manifest = _refresh_manifest_environments(manifest, root, built_env=built_env)

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
    elif not config.is_maintainer and not quiet:
        print("Manifest saved locally (not pushing - you are not the maintainer).")

    return True  # Not maintainer or no API key = skip push (not an error)


def _refresh_manifest_environments(
    manifest: Manifest, root: Path, built_env: str | None = None
) -> Manifest:
    """
    Update manifest with current environment state.

    Scans built environments and updates their info in the manifest.

    built_at semantics: `built_env` (the env the calling command just built)
    is stamped now; envs already in the manifest keep their recorded time; an
    env the manifest has never seen gets the env directory's mtime as a
    best-effort estimate — never `now`, which would fake freshness into the
    `verified_at > built_at` staleness comparison.
    """
    from .. import __version__
    from ..environment import list_built_environments

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

        # Hash the build's lockfile if the env has one (envs built before
        # lockfiles existed won't)
        lock_file = env_path / "env_source.py.lock"
        lock_hash = compute_source_hash(lock_file) if lock_file.exists() else None

        # Get python requires from source
        python_requires = get_requires_python(source_file) or ">=3.11"

        # Get direct dependencies from source
        direct_deps = get_dependencies(source_file)
        # Always track rootstock itself
        if "rootstock" not in [d.lower() for d in direct_deps]:
            direct_deps.append("rootstock")

        # Get installed package versions (filtered to direct dependencies)
        dependencies = get_installed_versions(env_path, only_packages=direct_deps)

        # Get checkpoints (from existing manifest if available)
        existing_env = manifest.environments.get(env_name)
        checkpoints = existing_env.checkpoints if existing_env else {}

        if env_name == built_env:
            built_at = now_iso()
        elif existing_env:
            built_at = existing_env.built_at
        else:
            built_at = built_at_estimate(env_path)

        manifest.environments[env_name] = EnvironmentInfo(
            built_at=built_at,
            source_hash=source_hash,
            source=source_content,
            python_requires=python_requires,
            dependencies=dependencies,
            checkpoints=checkpoints,
            lock_hash=lock_hash,
        )

    return manifest


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
        print(json.dumps(manifest.to_dict(), indent=2))
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
            print(f"      Built at:     {env.built_at}")
            print(f"      Source hash:  {env.source_hash[:20]}...")
            lock_desc = f"{env.lock_hash[:20]}..." if env.lock_hash else "none (pre-lockfile build)"
            print(f"      Lockfile:     {lock_desc}")
            print(f"      Dependencies: {len(env.dependencies)} packages")
            if env.checkpoints:
                print(f"      Checkpoints:  {', '.join(env.checkpoints.keys())}")

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
            "Configure API credentials in ~/.config/rootstock/config.toml",
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
        print(
            "Set maintainer info in ~/.config/rootstock/config.toml or run 'rootstock init'.",
            file=sys.stderr,
        )

    # Create and save manifest
    manifest = create_manifest(root, cluster, config)
    manifest = _refresh_manifest_environments(manifest, root)
    save_manifest(manifest, root)

    print(f"Manifest initialized: {root}/manifest.json")
    print(f"  Cluster: {cluster}")
    print(f"  Environments: {len(manifest.environments)}")

    # Skip push if explicitly disabled
    if getattr(args, "no_push", False):
        print("Manifest saved locally (push disabled via --no-push).")
        return 0

    # Push if configured AND user is maintainer
    if config.is_maintainer and config.is_push_enabled():
        client = RootstockClient(config)
        success, message = client.push_manifest(manifest)
        if success:
            print(f"Manifest pushed: {message}")
        else:
            print(f"Warning: Failed to push manifest: {message}", file=sys.stderr)
            print("Run 'rootstock manifest push' to retry.", file=sys.stderr)
    elif not config.is_maintainer:
        print("Manifest saved locally (not pushing - you are not the maintainer).")

    return 0
