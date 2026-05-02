"""Interactive initialization command."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ..clusters import CLUSTER_REGISTRY, get_cluster_for_root
from ..config import DEFAULT_CONFIG_FILE, load_config, save_config
from ..manifest import create_manifest, save_manifest
from .common import ROOTSTOCK_ROOT_ENV
from .manifest import _refresh_manifest_environments


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
        root = CLUSTER_REGISTRY[root_input].root
        print(f"  -> Using cluster '{cluster}' root: {root}")
    else:
        root = Path(root_input).expanduser().resolve()
        cluster = get_cluster_for_root(root)
        if cluster:
            print(f"  -> Detected cluster: {cluster}")

    config.root = str(root)

    print()

    # Ask if user is the maintainer
    print("Are you the maintainer of this rootstock installation?")
    print("Maintainers can configure API credentials to push manifests to the backend.")
    is_maintainer_input = prompt_with_default("Maintainer (y/n)", "n")
    config.is_maintainer = is_maintainer_input.lower() in ("y", "yes")

    print()

    if config.is_maintainer:
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
    else:
        print("Skipping maintainer and API configuration.")

    # Save configuration
    save_config(config)
    print(f"Configuration saved to {DEFAULT_CONFIG_FILE}")

    # Create directory structure. Most dirs live under the install root,
    # but cache/ and home/ live under the cache root — these may be the same
    # path or different filesystems depending on the cluster.
    if not args.skip_dirs:
        print("\nCreating directory structure...")
        from ..clusters import get_cluster
        if cluster:
            cache_root = get_cluster(cluster).resolved_cache_root
        else:
            cache_root = root

        dirs_to_create = [
            root / "environments",
            root / "envs",
            root / ".python",
            cache_root / "cache",
            cache_root / "home",
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
        # Scan for existing built environments
        manifest = _refresh_manifest_environments(manifest, root)
        save_manifest(manifest, root)
        print(f"  Created: {root}/manifest.json")
        if manifest.environments:
            print(f"  Found {len(manifest.environments)} existing environment(s)")

        # Push if configured AND user is maintainer
        if config.is_maintainer and config.is_push_enabled():
            from ..client import RootstockClient

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
