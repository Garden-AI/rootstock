"""``rootstock smoke-test`` — re-verify checkpoints already in the manifest."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..local_checkpoints import (
    LocalCheckpointEntry,
    LocalCheckpointError,
    hash_weights_file,
    local_checkpoints_for_root,
    record_local_verification,
)
from ..manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    is_verified,
    load_manifest,
    manifest_lock,
    now_iso,
    save_manifest,
)
from ..operations import update_and_push_manifest
from ..verify import verify_checkpoint
from .common import get_root_or_exit, resolve_cache_root


def _select(
    manifest, env_filter: str | None, checkpoint_filter: str | None
) -> list[tuple[str, str, EnvironmentInfo, CheckpointInfo]]:
    """Pick which (env, checkpoint) pairs to test."""
    selected: list[tuple[str, str, EnvironmentInfo, CheckpointInfo]] = []
    for env_name, env in manifest.environments.items():
        if env_filter is not None and env_name != env_filter:
            continue
        for ckpt_name, ckpt in env.checkpoints.items():
            if checkpoint_filter is not None and ckpt_name != checkpoint_filter:
                continue
            if ckpt.fetched_at is None:
                # Smoke-test never downloads. Skip checkpoints that have never been fetched.
                continue
            selected.append((env_name, ckpt_name, env, ckpt))
    return selected


def _select_local(
    root: Path, env_filter: str | None, checkpoint_filter: str | None
) -> list[tuple[str, LocalCheckpointEntry]]:
    """Pick which of the user's local checkpoints to test (same filter
    semantics as the manifest selection; --env matches the hosting env)."""
    try:
        local = local_checkpoints_for_root(root)
    except LocalCheckpointError as exc:
        print(f"Warning: ignoring local-checkpoint registry: {exc}", file=sys.stderr)
        return []
    selected = []
    for ckpt_id, entry in sorted(local.items()):
        if env_filter is not None and entry.env != env_filter:
            continue
        if checkpoint_filter is not None and ckpt_id != checkpoint_filter:
            continue
        selected.append((ckpt_id, entry))
    return selected


def _check_local_weights(entry: LocalCheckpointEntry) -> str | None:
    """Pre-verify integrity check: the file must still exist and hash to what
    was registered — a silently swapped file must never be blessed."""
    path = Path(entry.path)
    if not path.exists():
        return f"weights file missing: {entry.path}"
    sha256, size = hash_weights_file(path)
    if sha256 != entry.sha256 or size != entry.size:
        return (
            "weights file changed on disk (sha256 mismatch); "
            "re-register with `rootstock add-local`"
        )
    return None


def cmd_smoke_test(args) -> int:
    root: Path = get_root_or_exit(args)
    cache_root = resolve_cache_root(root)
    env_filter = args.env
    ckpt_filter = args.checkpoint
    device = args.device
    json_out = args.json
    no_push = args.no_push

    if ckpt_filter is not None and env_filter is None:
        print("Error: --checkpoint requires --env", file=sys.stderr)
        return 2

    local_selected = _select_local(root, env_filter, ckpt_filter)

    manifest = load_manifest(root)
    if manifest is None and not local_selected:
        print(f"Error: no manifest at {root}/manifest.json", file=sys.stderr)
        return 1

    selected = _select(manifest, env_filter, ckpt_filter) if manifest is not None else []
    if not selected and not local_selected:
        if json_out:
            print(json.dumps({"results": [], "passed": 0, "failed": 0}, indent=2))
        else:
            print("No fetched checkpoints to test.")
        return 0

    results: list[dict] = []
    n_passed = 0
    n_failed = 0
    total_start = time.monotonic()

    # Verification runs are long (minutes per checkpoint) and must not hold
    # the manifest lock. Record outcomes here and apply them to a *freshly
    # loaded* manifest in one locked cycle afterwards — mutating the manifest
    # loaded before the loop and saving it at the end would silently revert
    # anything a co-maintainer wrote in between.
    outcomes: list[tuple[str, str, bool, str | None]] = []

    for env_name, ckpt_name, env, ckpt in selected:
        start = time.monotonic()
        ok, err = verify_checkpoint(
            root=root,
            env_name=env_name,
            checkpoint=ckpt_name,
            device=device,
            setup_kwargs={},  # smoke-test always uses env defaults; see design §7.2
            cache_root=cache_root,
        )
        elapsed = time.monotonic() - start
        outcomes.append((env_name, ckpt_name, ok, err))

        # Mirror the outcome onto the working copy so verified_current in the
        # report reflects this run.
        if ok:
            ckpt.verified_at = now_iso()
            ckpt.verified_device = device
            ckpt.last_error = None
            n_passed += 1
        else:
            ckpt.verified_at = None
            ckpt.verified_device = None
            ckpt.last_error = f"smoke-test: {err}"
            n_failed += 1

        results.append(
            {
                "env": env_name,
                "checkpoint": ckpt_name,
                "device": device,
                "passed": ok,
                "elapsed_s": round(elapsed, 2),
                "error": err,
                "verified_current": is_verified(env, ckpt),
            }
        )

        if not json_out:
            verdict = "PASS" if ok else "FAIL"
            line = f"{env_name}/{ckpt_name:<24} [{verdict}]  {device}  {elapsed:5.1f}s"
            if not ok:
                line += f"  {err}"
            print(line)

    # Local checkpoints: re-hash first (a swapped file must not be blessed),
    # then verify with the *registered* kwargs — unlike canonical smoke-tests
    # they are part of the registration contract, and they're what add-local
    # verified. Outcomes go to the per-user registry, never the manifest.
    local_outcomes: list[tuple[str, bool, str | None]] = []
    for ckpt_id, entry in local_selected:
        start = time.monotonic()
        err = _check_local_weights(entry)
        if err is None:
            ok, err = verify_checkpoint(
                root=root,
                env_name=entry.env,
                checkpoint=ckpt_id,
                device=device,
                setup_kwargs=entry.setup_kwargs,
                cache_root=cache_root,
                checkpoint_path=entry.path,
            )
        else:
            ok = False
        elapsed = time.monotonic() - start
        local_outcomes.append((ckpt_id, ok, err))

        if ok:
            n_passed += 1
        else:
            n_failed += 1

        env_record = manifest.environments.get(entry.env) if manifest is not None else None
        results.append(
            {
                "env": entry.env,
                "checkpoint": ckpt_id,
                "device": device,
                "passed": ok,
                "elapsed_s": round(elapsed, 2),
                "error": err,
                "local": True,
                "verified_current": bool(
                    ok and env_record is not None and now_iso() > env_record.built_at
                ),
            }
        )

        if not json_out:
            verdict = "PASS" if ok else "FAIL"
            line = (
                f"{entry.env}/{ckpt_id:<24} [{verdict}]  {device}  "
                f"{elapsed:5.1f}s  [local]"
            )
            if not ok:
                line += f"  {err}"
            print(line)

    for ckpt_id, ok, err in local_outcomes:
        # record_local_verification loads fresh and skips ids removed while
        # we were testing; the shared manifest is never touched for locals.
        record_local_verification(
            root,
            ckpt_id,
            ok=ok,
            device=device,
            error=None if ok else f"smoke-test: {err}",
        )

    # Only touch the shared manifest when canonical checkpoints were tested.
    # A local-only run (the common case for non-maintainers, who cannot write
    # the shared root at all) must not take the manifest lock or push.
    if outcomes:
        with manifest_lock(root):
            fresh = load_manifest(root)
            if fresh is not None:
                for env_name, ckpt_name, ok, err in outcomes:
                    env_record = fresh.environments.get(env_name)
                    ckpt_record = env_record.checkpoints.get(ckpt_name) if env_record else None
                    if ckpt_record is None:
                        continue  # env/checkpoint removed while we were testing
                    if ok:
                        ckpt_record.verified_at = now_iso()
                        ckpt_record.verified_device = device
                        ckpt_record.last_error = None
                    else:
                        ckpt_record.verified_at = None
                        ckpt_record.verified_device = None
                        ckpt_record.last_error = f"smoke-test: {err}"
                save_manifest(fresh, root)
        update_and_push_manifest(root, quiet=True, push=not no_push)

    total_elapsed = time.monotonic() - total_start

    if json_out:
        print(
            json.dumps(
                {
                    "results": results,
                    "passed": n_passed,
                    "failed": n_failed,
                    "elapsed_s": round(total_elapsed, 2),
                },
                indent=2,
            )
        )
    else:
        print(f"\n{n_passed} passed, {n_failed} failed in {total_elapsed:.1f}s")

    return 0 if n_failed == 0 else 1
