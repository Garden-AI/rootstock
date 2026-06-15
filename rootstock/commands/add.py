"""``rootstock add`` — idempotent download-or-verify for a checkpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..environment import (
    CheckpointNotFoundError,
    find_env_for_checkpoint,
    get_model_cache_env,
    list_declared_checkpoints,
)
from ..manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    Manifest,
    create_manifest,
    load_manifest,
    now_iso,
    save_manifest,
)
from ..verify import verify_checkpoint
from .common import get_root_or_exit, resolve_cache_root
from .manifest import update_and_push_manifest


def parse_kwarg(spec: str) -> tuple[str, object]:
    """
    Parse a ``key=val`` CLI arg. Value is JSON-decoded first; on parse
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


def _kwargs_from_args(kwarg_specs: list[str] | None) -> dict[str, object]:
    out: dict[str, object] = {}
    for spec in kwarg_specs or []:
        key, value = parse_kwarg(spec)
        out[key] = value
    return out


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
    env_python = env_dir / "bin" / "python"
    if not env_python.exists():
        return False, f"environment not built at {env_dir}"

    fd, kwargs_path = tempfile.mkstemp(suffix=".json", prefix="rootstock_add_kwargs_")
    try:
        with open(fd, "w") as f:
            json.dump(setup_kwargs, f)

        script = (
            "import sys, json\n"
            f'sys.path.insert(0, "{env_dir}")\n'
            "from env_source import setup\n"
            f'with open("{kwargs_path}") as _f: kwargs = json.load(_f)\n'
            f'setup({checkpoint!r}, "cpu", **kwargs)\n'
        )

        env = {**os.environ, **get_model_cache_env(root, cache_root)}
        # Run from env_dir so the implicit "" entry that `python -c` puts on
        # sys.path resolves to the env directory (which holds only env_source.py
        # and the venv internals) rather than the caller's CWD. Without this, a
        # config whose top-level import name matches a file in the caller's CWD
        # — e.g. running from environments/ where mace.py lives while adding a
        # checkpoint whose setup() does `import mace` — shadows the installed
        # package and fails with "'mace' is not a package".
        result = subprocess.run(
            [str(env_python), "-c", script],
            env=env,
            cwd=str(env_dir),
            capture_output=True,
            text=True,
        )
    finally:
        try:
            Path(kwargs_path).unlink()
        except OSError:
            pass

    if result.returncode != 0:
        err = result.stderr.strip().splitlines()
        # Last line is usually the most informative (the actual exception message).
        msg = err[-1] if err else f"setup() exited with code {result.returncode}"
        return False, msg
    return True, None


def _ensure_manifest_entry(
    root: Path,
    cluster_hint: str | None,
    env_name: str,
    checkpoint: str,
) -> tuple[Manifest, EnvironmentInfo, CheckpointInfo]:
    """Load (or create) the manifest and return the env+checkpoint records."""
    from ..config import load_config

    manifest = load_manifest(root)
    if manifest is None:
        config = load_config()
        cluster = cluster_hint or "unknown"
        manifest = create_manifest(root, cluster, config)

    env = manifest.environments.get(env_name)
    if env is None:
        # The env directory must exist for the operation to make sense, but if
        # the manifest hasn't been refreshed yet we'll synthesize a minimal
        # entry. update_and_push_manifest() will fill in the rest on save.
        env_dir = root / "envs" / env_name
        if not (env_dir / "bin" / "python").exists():
            raise RuntimeError(
                f"environment '{env_name}' is not built at {env_dir}.\n"
                f"Run: rootstock install <path-to-{env_name}.py> --root {root}"
            )
        env = EnvironmentInfo(
            status="ready",
            built_at=now_iso(),
            source_hash="",
            source="",
            python_requires=">=3.10",
            dependencies={},
            checkpoints={},
        )
        manifest.environments[env_name] = env

    if checkpoint not in env.checkpoints:
        env.checkpoints[checkpoint] = CheckpointInfo()
    return manifest, env, env.checkpoints[checkpoint]


def _print_checkpoint_catalog(root: Path) -> int:
    """Print every canonical checkpoint id ``rootstock add`` accepts, grouped
    by hosting env. Returns a process exit code."""
    declared = list_declared_checkpoints(root)
    if not declared:
        print(
            f"No envs are installed at {root}. "
            f"Run `rootstock install <env-file> --root {root}` first."
        )
        return 0

    print(f"Checkpoints available to add in {root}:")
    for env_name, ckpts in declared.items():
        print(f"  {env_name}:")
        if not ckpts:
            print("    (none)")
            continue
        for ckpt_id in ckpts:
            print(f"    {ckpt_id}")
    return 0


def cmd_add(args) -> int:
    root = get_root_or_exit(args)

    if getattr(args, "list", False):
        return _print_checkpoint_catalog(root)

    if not args.checkpoint:
        print(
            "Error: a checkpoint id is required (or pass --list to see available ids)",
            file=sys.stderr,
        )
        return 2

    cache_root = resolve_cache_root(root)
    checkpoint = args.checkpoint
    device = args.device
    no_verify = args.no_verify
    no_push = args.no_push

    try:
        kwargs = _kwargs_from_args(args.kwarg)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        env_name, _ = find_env_for_checkpoint(root, checkpoint)
    except CheckpointNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        manifest, env, ckpt = _ensure_manifest_entry(root, None, env_name, checkpoint)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # ---- Download phase ------------------------------------------------
    if ckpt.fetched_at is None:
        print(f"Downloading {env_name}/{checkpoint} on CPU...")
        ok, err = _run_download(root, env_name, checkpoint, kwargs, cache_root=cache_root)
        if not ok:
            ckpt.last_error = f"download: {err}"
            save_manifest(manifest, root)
            print(f"Error: download failed: {err}", file=sys.stderr)
            return 1
        ckpt.fetched_at = now_iso()
        ckpt.last_error = None
        save_manifest(manifest, root)
        print(f"  fetched_at = {ckpt.fetched_at}")
    else:
        print(f"{env_name}/{checkpoint} already fetched at {ckpt.fetched_at}")

    # ---- Verify phase --------------------------------------------------
    if no_verify:
        print("(skipping verify per --no-verify)")
    else:
        print(f"Verifying {env_name}/{checkpoint} on {device}...")
        ok, err = verify_checkpoint(
            root, env_name, checkpoint, device, kwargs, cache_root=cache_root
        )
        if not ok:
            ckpt.verified_at = None
            ckpt.verified_device = None
            ckpt.last_error = f"verify: {err}"
            save_manifest(manifest, root)
            print(f"Error: verify failed: {err}", file=sys.stderr)
            return 1
        ckpt.verified_at = now_iso()
        ckpt.verified_device = device
        ckpt.last_error = None
        save_manifest(manifest, root)
        print(f"  verified_at = {ckpt.verified_at} ({device})")

    # Refresh + push
    update_and_push_manifest(root, quiet=True, push=not no_push)
    return 0
