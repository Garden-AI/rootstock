"""Status and list commands."""

from __future__ import annotations

import json

from ..config import DEFAULT_CONFIG_FILE
from ..manifest import is_verified, load_manifest
from .common import get_root_or_exit, resolve_cache_root


def _short_date(iso: str | None) -> str:
    if not iso:
        return "—"
    return iso[:10]


def _checkpoint_line(env, ckpt_name: str, ckpt) -> str:
    fetched = f"fetched {_short_date(ckpt.fetched_at)}"
    if ckpt.last_error and ckpt.fetched_at is None:
        # Never successfully fetched.
        return f"    {ckpt_name:<24}  not fetched   ⚠  {ckpt.last_error}"

    if ckpt.verified_at is None:
        verified = "not verified"
        marker = "⚠"
    elif is_verified(env, ckpt):
        verified = f"verified {_short_date(ckpt.verified_at)} ({ckpt.verified_device})"
        marker = "✓"
    else:
        verified = (
            f"verified {_short_date(ckpt.verified_at)} ({ckpt.verified_device})  "
            f"⚠ stale (env rebuilt {_short_date(env.built_at)})"
        )
        marker = ""

    line = f"    {ckpt_name:<24}  {fetched}  {verified}  {marker}".rstrip()
    if ckpt.last_error:
        line += f"\n      last error: {ckpt.last_error}"
    return line


def cmd_status(args) -> int:
    """Show status of rootstock installation."""
    from ..environment import list_built_environments, list_environments

    root = get_root_or_exit(args)
    manifest = load_manifest(root)

    if getattr(args, "json", False):
        return _cmd_status_json(root, manifest)

    print(f"Rootstock root: {root}")

    # List environment sources
    print("\nEnvironment sources:")
    sources = list_environments(root)
    if not sources:
        print("  (none)")
    else:
        for name, path in sources:
            print(f"  {name}")

    # List built environments + per-checkpoint verification state
    print("\nBuilt environments:")
    built = list_built_environments(root)
    if not built:
        print("  (none)")
    else:
        for name, path in built:
            has_source = (path / "env_source.py").exists()
            status = "ready" if has_source else "incomplete"
            print(f"  {name:<20} [{status}]")

            env = manifest.environments.get(name) if manifest else None
            if env is None:
                continue
            if not env.checkpoints:
                print("    (no checkpoints — run 'rootstock add <checkpoint-id>')")
                continue
            print(f"    Built: {env.built_at}")
            print(f"    Checkpoints ({len(env.checkpoints)}):")
            for ckpt_name, ckpt in env.checkpoints.items():
                print(_checkpoint_line(env, ckpt_name, ckpt))

    # Show cache sizes. Cache may live under the install root or under a
    # separate cluster-registered cache_root. Some libraries respect
    # XDG_CACHE_HOME and write under cache/; others hardcode ~/.cache/ or
    # ~/.matgl/ etc. and write under our redirected home/. Sum both.
    cache_root = resolve_cache_root(root)
    print(f"\nCache: {cache_root}")

    locations: list = []
    for parent in (cache_root / "cache", cache_root / "home" / ".cache"):
        if parent.exists():
            locations.extend(p for p in sorted(parent.iterdir()) if p.is_dir())
    home_dir = cache_root / "home"
    if home_dir.exists():
        for sub in sorted(home_dir.iterdir()):
            if sub.is_dir() and sub.name != ".cache":
                locations.append(sub)

    if not locations:
        print("  (empty)")
    else:
        total_bytes = 0
        for loc in locations:
            size = sum(f.stat().st_size for f in loc.rglob("*") if f.is_file())
            total_bytes += size
            rel = loc.relative_to(cache_root)
            print(f"  {str(rel) + '/':<32} {size / (1024 * 1024):>8.1f} MB")
        print(f"  {'TOTAL':<32} {total_bytes / (1024 * 1024):>8.1f} MB")

    # Show config file location
    print(f"\nConfig file: {DEFAULT_CONFIG_FILE}")

    return 0


def _cmd_status_json(root, manifest) -> int:
    """Emit raw manifest data plus computed verified_current per checkpoint."""
    if manifest is None:
        print(json.dumps({"root": str(root), "manifest": None}))
        return 0

    data = manifest.to_dict()
    for env_name, env in manifest.environments.items():
        env_data = data["environments"][env_name]
        for ckpt_name, ckpt in env.checkpoints.items():
            env_data["checkpoints"][ckpt_name]["verified_current"] = is_verified(env, ckpt)
    print(json.dumps({"root": str(root), "manifest": data}, indent=2))
    return 0


def cmd_list(args) -> int:
    """List registered environments."""
    from ..environment import list_built_environments, list_environments

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
