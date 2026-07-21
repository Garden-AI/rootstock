"""Status and list commands."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import DEFAULT_CONFIG_FILE
from ..install_state import InstallState, read_install_state
from ..local_checkpoints import LocalCheckpointError, local_checkpoints_for_root
from ..manifest import is_verified
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


def _local_state(state: InstallState, ckpt_id: str, entry) -> dict:
    """Derived flags for one local checkpoint: file presence, staleness
    against the hosting env's manifest built_at, and canonical shadowing."""
    env = state.envs.get(entry.env)
    record = env.record if env is not None else None

    shadowed_by = None
    for env_name, env_state in state.envs.items():
        if env_state.declared_checkpoints and ckpt_id in env_state.declared_checkpoints:
            shadowed_by = env_name
            break

    if entry.verified_at is None:
        verified_current = False
    elif record is None:
        verified_current = False
    else:
        # Same lexical-ISO comparison as manifest.is_verified.
        verified_current = entry.verified_at > record.built_at

    return {
        "exists": Path(entry.path).exists(),
        "env_built": env is not None,
        "env_built_at": record.built_at if record else None,
        "verified_current": verified_current,
        "shadowed_by": shadowed_by,
    }


def _local_checkpoint_line(state: InstallState, ckpt_id: str, entry) -> str:
    derived = _local_state(state, ckpt_id, entry)

    if entry.verified_at is None:
        verified = "not verified"
        marker = "⚠"
    elif derived["verified_current"]:
        verified = f"verified {_short_date(entry.verified_at)} ({entry.verified_device})"
        marker = "✓"
    elif derived["env_built_at"] is not None:
        verified = (
            f"verified {_short_date(entry.verified_at)} ({entry.verified_device})  "
            f"⚠ stale (env rebuilt {_short_date(derived['env_built_at'])})"
        )
        marker = ""
    else:
        verified = (
            f"verified {_short_date(entry.verified_at)} ({entry.verified_device})  "
            f"⚠ env not in manifest"
        )
        marker = ""

    line = f"    {ckpt_id:<24}  env: {entry.env:<12}  {verified}  {marker}".rstrip()
    line += f"\n      {entry.path}"
    if not derived["exists"]:
        line += "  ⚠ file missing"
    if not derived["env_built"]:
        line += f"\n      ⚠ env '{entry.env}' is not built"
    if derived["shadowed_by"]:
        line += (
            f"\n      ⚠ shadowed by a canonical id in env "
            f"'{derived['shadowed_by']}' — the canonical checkpoint wins"
        )
    if entry.last_error:
        line += f"\n      last error: {entry.last_error}"
    return line


def cmd_status(args) -> int:
    """Show status of rootstock installation."""
    root = get_root_or_exit(args)
    state = read_install_state(root)

    if getattr(args, "json", False):
        return _cmd_status_json(state)

    print(f"Rootstock root: {root}")

    # List environment sources
    print("\nEnvironment sources:")
    if not state.sources:
        print("  (none)")
    else:
        for name, path in state.sources:
            print(f"  {name}")

    # List built environments + per-checkpoint verification state
    print("\nBuilt environments:")
    if not state.envs and not state.manifest_only_envs:
        print("  (none)")
    else:
        for name, env in state.envs.items():
            status = "ready" if env.source_file is not None else "incomplete"
            print(f"  {name:<20} [{status}]")

            if env.source_hash_drifted:
                print(
                    "    ⚠ env_source.py on disk differs from the manifest "
                    "(in-place hotfix?) — run 'rootstock smoke-test' to re-record"
                )

            if env.record is None:
                continue
            if not env.record.checkpoints:
                print("    (no checkpoints — run 'rootstock add <checkpoint-id>')")
                continue
            print(f"    Built: {env.record.built_at}")
            print(f"    Checkpoints ({len(env.record.checkpoints)}):")
            for ckpt_name, ckpt in env.record.checkpoints.items():
                print(_checkpoint_line(env.record, ckpt_name, ckpt))

        # Records the manifest still carries for envs that are gone from disk.
        for name in sorted(state.manifest_only_envs):
            print(f"  {name:<20} [manifest only — not on disk]")

    # The user's own registered local checkpoints (per-user registry; never
    # part of the shared manifest).
    try:
        local = local_checkpoints_for_root(root)
        local_error = None
    except LocalCheckpointError as exc:
        local, local_error = {}, str(exc)
    if local or local_error:
        print("\nLocal checkpoints (this user):")
        if local_error:
            print(f"  (unreadable: {local_error})")
        for ckpt_id, entry in sorted(local.items()):
            print(_local_checkpoint_line(state, ckpt_id, entry))

    # Show the cache location; computing sizes is opt-in. Cache may live
    # under the install root or under a separate declared cache_root. Some
    # libraries respect XDG_CACHE_HOME and write under cache/; others
    # hardcode ~/.cache/ or ~/.matgl/ etc. and write under our redirected
    # home/ — sizing sums both, which means a full rglob+stat over the model
    # cache: minutes of metadata traffic on Lustre/GPFS for big HF caches,
    # so it only runs with --sizes.
    cache_root = resolve_cache_root(root)
    print(f"\nCache: {cache_root}")

    if not getattr(args, "sizes", False):
        print("  (pass --sizes to compute per-directory cache sizes)")
        print(f"\nConfig file: {DEFAULT_CONFIG_FILE}")
        return 0

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


def _cmd_status_json(state: InstallState) -> int:
    """Emit the merged install state as JSON.

    Environments are enumerated from the filesystem (the truth for what's
    installed); manifest metadata (built_at, checkpoint fetch/verify state)
    is joined in where it exists. Manifest records for envs no longer on
    disk are listed separately rather than passed off as installed.
    """
    environments = {}
    for name, env in state.envs.items():
        checkpoints = {}
        if env.record is not None:
            for ckpt_name, ckpt in env.record.checkpoints.items():
                ckpt_data = ckpt.to_dict()
                ckpt_data["verified_current"] = is_verified(env.record, ckpt)
                checkpoints[ckpt_name] = ckpt_data
        environments[name] = {
            "has_source": env.source_file is not None,
            "source_hash": env.source_hash,
            "lock_hash": env.lock_hash,
            "declared_checkpoints": sorted(env.declared_checkpoints or {}),
            "in_manifest": env.record is not None,
            "built_at": env.record.built_at if env.record else None,
            "manifest_source_hash": env.record.source_hash if env.record else None,
            "checkpoints": checkpoints,
        }

    local_checkpoints = {}
    try:
        for ckpt_id, entry in sorted(local_checkpoints_for_root(state.root).items()):
            local_checkpoints[ckpt_id] = {
                **entry.to_dict(),
                **_local_state(state, ckpt_id, entry),
            }
    except LocalCheckpointError:
        pass  # per-user registry problems must not break machine-readable status

    manifest = state.manifest
    payload = {
        "root": str(state.root),
        "cluster": manifest.cluster if manifest else None,
        "maintainer": manifest.maintainer.to_dict() if manifest else None,
        "rootstock_version": manifest.rootstock_version if manifest else None,
        "last_updated": manifest.last_updated if manifest else None,
        "sources": [name for name, _ in state.sources],
        "environments": environments,
        "manifest_only_environments": sorted(state.manifest_only_envs),
        "local_checkpoints": local_checkpoints,
    }
    print(json.dumps(payload, indent=2))
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
