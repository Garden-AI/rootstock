"""``rootstock smoke-test`` — re-verify checkpoints already in the manifest."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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

    manifest = load_manifest(root)
    if manifest is None:
        print(f"Error: no manifest at {root}/manifest.json", file=sys.stderr)
        return 1

    selected = _select(manifest, env_filter, ckpt_filter)
    if not selected:
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
