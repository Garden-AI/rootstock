"""Manifest management commands."""

from __future__ import annotations

import json
import sys

from ..client import RootstockClient
from ..config import load_config
from ..manifest import create_manifest, load_manifest, save_manifest
from ..operations import refresh_manifest_environments
from .common import get_root_or_exit


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
    manifest = refresh_manifest_environments(manifest, root)
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

    return 0
