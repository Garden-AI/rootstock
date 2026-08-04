"""``rootstock smoke-test`` — re-verify checkpoints already in the manifest.

Besides re-verifying every fetched canonical checkpoint, each run exercises
the custom-weights path (#200): for every ``<family>:custom`` entry an env
declares, the weights file a same-family canonical checkpoint already has on
disk is loaded again via ``weights=`` (``setup_from_path``), and the two
calculators must agree on the smoke-test structure. The outcome is recorded
on the ``<family>:custom`` manifest entry so per-cluster support for
user-supplied weights is visible downstream (e.g. on the almanac).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..environment import (
    CUSTOM_CHECKPOINT_SUFFIX,
    is_custom_checkpoint,
    parse_checkpoints_dict,
    parse_clusters_list,
    parse_custom_checkpoint_ids,
)
from ..manifest import (
    CheckpointInfo,
    EnvironmentInfo,
    Manifest,
    VerificationRecord,
    is_verified,
    load_manifest,
    manifest_lock,
    now_iso,
    save_manifest,
)
from ..operations import (
    _WEIGHTS_BYTE_FLOOR,
    OperationError,
    apply_weights_record,
    read_weights_capture,
    resolve_current_cluster,
    update_and_push_manifest,
    weights_capture_file,
)
from ..verify import results_mismatch, verify_checkpoint
from .common import get_root_or_exit, resolve_cache_root


def _serves_here(root: Path, env_name: str, env: EnvironmentInfo, cluster: str) -> bool:
    """Whether an env serves the cluster this run is on: the live source's
    CLUSTERS when readable (so a fresh restriction takes effect this run, not
    after the next refresh), else the manifest record."""
    source = root / "envs" / env_name / "env_source.py"
    if source.exists():
        try:
            restriction = parse_clusters_list(source)
        except ValueError:
            return env.serves(cluster)
        return restriction is None or cluster in restriction
    return env.serves(cluster)


def _select(
    root: Path, manifest, env_filter: str | None, checkpoint_filter: str | None, cluster: str
) -> list[tuple[str, str, EnvironmentInfo, CheckpointInfo]]:
    """Pick which (env, checkpoint) pairs to test."""
    selected: list[tuple[str, str, EnvironmentInfo, CheckpointInfo]] = []
    for env_name, env in manifest.environments.items():
        if env_filter is not None and env_name != env_filter:
            continue
        if not _serves_here(root, env_name, env, cluster):
            # A cluster-restricted variant can only be verified by a machine
            # it serves (#208); its own cluster's chain covers it.
            continue
        for ckpt_name, ckpt in env.checkpoints.items():
            if checkpoint_filter is not None and ckpt_name != checkpoint_filter:
                continue
            if is_custom_checkpoint(ckpt_name):
                # ':custom' records track the weights= leg below; they carry
                # no shipped weights for the canonical loop to verify.
                continue
            if ckpt.fetched_at is None:
                # Smoke-test never downloads. Skip checkpoints that have never been fetched.
                continue
            selected.append((env_name, ckpt_name, env, ckpt))
    return selected


@dataclass(frozen=True)
class CustomLeg:
    """One planned run of the weights= path: re-load ``base``'s cached
    weights file under the ``custom_id`` entry and compare results."""

    env_name: str
    custom_id: str
    base: str  # same-family canonical checkpoint whose weights drive the leg


def _family_of(ckpt_id: str, families: list[str]) -> str | None:
    """The family a canonical id belongs to: the longest declared family that
    equals it or prefixes it up to a ``-`` or a version digit — so
    'mace-off23-small' belongs to 'mace-off' (not 'mace'), 'orb-v2' to
    'orb-v2', while 'macex-1' belongs to no family named 'mace'."""
    best = None
    for family in families:
        matches = ckpt_id == family or (
            ckpt_id.startswith(family) and ckpt_id[len(family)] in "-0123456789"
        )
        if matches and (best is None or len(family) > len(best)):
            best = family
    return best


def _plan_custom_legs(
    root: Path,
    manifest: Manifest,
    env_filter: str | None,
    ckpt_filter: str | None,
    cluster: str,
) -> tuple[list[CustomLeg], list[tuple[str, str, str]]]:
    """Plan one weights= leg per ``<family>:custom`` entry declared by a
    manifest env's built source (#200).

    The base is the first canonical id (in CHECKPOINTS declaration order —
    env authors lead with the family's plainest-loading checkpoint) of the
    same family that the manifest records as fetched. Returns the legs plus
    ``(env, custom_id, reason)`` for entries that can't run this time.
    """
    legs: list[CustomLeg] = []
    skipped: list[tuple[str, str, str]] = []
    for env_name, env in manifest.environments.items():
        if env_filter is not None and env_name != env_filter:
            continue
        if not _serves_here(root, env_name, env, cluster):
            continue
        source = Path(root) / "envs" / env_name / "env_source.py"
        if not source.exists():
            continue
        try:
            declared = parse_checkpoints_dict(source)
            custom_ids = parse_custom_checkpoint_ids(source)
        except ValueError:
            continue
        families = [c.removesuffix(CUSTOM_CHECKPOINT_SUFFIX) for c in custom_ids]
        fetched = [
            ckpt_id
            for ckpt_id in declared
            if (record := env.checkpoints.get(ckpt_id)) and record.fetched_at
        ]
        for custom_id, family in zip(custom_ids, families):
            if ckpt_filter is not None and ckpt_filter != custom_id:
                continue
            base = next((c for c in fetched if _family_of(c, families) == family), None)
            if base is None:
                skipped.append(
                    (
                        env_name,
                        custom_id,
                        f"no fetched '{family}' checkpoint to borrow weights from",
                    )
                )
                continue
            legs.append(CustomLeg(env_name, custom_id, base))
    return legs, skipped


def _dominant_weights_file(files: list[dict] | None) -> str | None:
    """The single weights file to hand to ``weights=``, from a load capture
    (#177): the largest recorded file, required to be at least weights-sized
    and at least twice everything else combined — sidecars (configs,
    tokenizers) are small, while a second *shard* is comparable to the first.
    Multi-shard checkpoints have no such file (setup_from_path takes one
    path), so those can't run this leg. Returns the cache-root-relative
    path, or None."""
    if not files:
        return None
    largest = max(files, key=lambda f: f["size"])
    rest = sum(f["size"] for f in files) - largest["size"]
    if largest["size"] < _WEIGHTS_BYTE_FLOOR or largest["size"] < 2 * rest:
        return None
    return largest["path"]


def cmd_smoke_test(args) -> int:
    root: Path = get_root_or_exit(args)
    cache_root = resolve_cache_root(root)
    env_filter = args.env
    ckpt_filter = args.checkpoint
    device = args.device
    verify_timeout = args.verify_timeout
    json_out = args.json
    no_push = args.no_push

    if ckpt_filter is not None and env_filter is None:
        print("Error: --checkpoint requires --env", file=sys.stderr)
        return 2

    manifest = load_manifest(root)
    if manifest is None:
        print(f"Error: no manifest at {root}/manifest.json", file=sys.stderr)
        return 1

    # Which machine is this? Results are recorded (and pushed) under this
    # cluster's name; on a shared install it must be given explicitly (#208).
    try:
        cluster = resolve_current_cluster(root, args.cluster)
    except OperationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    custom_legs, custom_skips = _plan_custom_legs(root, manifest, env_filter, ckpt_filter, cluster)

    selected = _select(root, manifest, env_filter, ckpt_filter, cluster)
    if ckpt_filter is not None and is_custom_checkpoint(ckpt_filter):
        # A ':custom' filter matches no canonical row; run just the base
        # checkpoint(s) the leg needs for its comparison.
        needed = {(leg.env_name, leg.base) for leg in custom_legs}
        selected = [
            t for t in _select(root, manifest, env_filter, None, cluster) if (t[0], t[1]) in needed
        ]

    if not selected and not custom_legs:
        if json_out:
            print(
                json.dumps(
                    {
                        "cluster": cluster,
                        "results": [],
                        "passed": 0,
                        "failed": 0,
                        "skipped": [
                            {"env": e, "checkpoint": c, "reason": r} for e, c, r in custom_skips
                        ],
                    },
                    indent=2,
                )
            )
        else:
            for env_name, custom_id, reason in custom_skips:
                print(f"{env_name}/{custom_id:<24} [SKIP]  {reason}")
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
    outcomes: list[tuple[str, str, bool, str | None, list[dict] | None]] = []

    # The custom legs reuse the canonical loop's work: its results are the
    # comparison baseline, and its fresh weight capture names the file on
    # disk that weights= re-loads.
    base_keys = {(leg.env_name, leg.base) for leg in custom_legs}
    baselines: dict[tuple[str, str], dict] = {}
    base_captures: dict[tuple[str, str], list[dict] | None] = {}
    passed_keys: set[tuple[str, str]] = set()

    for env_name, ckpt_name, env, ckpt in selected:
        start = time.monotonic()
        # Each pass also re-captures which weight files the load touched, so
        # per-checkpoint weight records self-heal nightly like verified_at
        # (#177) — and backfilling them on an existing install is just one
        # smoke-test run, no re-downloads.
        run_results: dict = {}
        with weights_capture_file() as capture_path:
            ok, err = verify_checkpoint(
                root=root,
                env_name=env_name,
                checkpoint=ckpt_name,
                device=device,
                setup_kwargs={},  # smoke-test always uses env defaults; see design §7.2
                cache_root=cache_root,
                weights_capture_path=str(capture_path),
                results=run_results,
                timeout=verify_timeout,
            )
            weight_files = read_weights_capture(capture_path)
        elapsed = time.monotonic() - start
        outcomes.append((env_name, ckpt_name, ok, err, weight_files))

        if (env_name, ckpt_name) in base_keys:
            baselines[(env_name, ckpt_name)] = run_results
            base_captures[(env_name, ckpt_name)] = weight_files

        # Mirror the outcome onto the working copy so verified_current in the
        # report reflects this run.
        if ok:
            passed_keys.add((env_name, ckpt_name))
            ckpt.verifications[cluster] = VerificationRecord(
                verified_at=now_iso(), verified_device=device
            )
            n_passed += 1
        else:
            ckpt.verifications[cluster] = VerificationRecord(last_error=f"smoke-test: {err}")
            n_failed += 1

        results.append(
            {
                "env": env_name,
                "checkpoint": ckpt_name,
                "cluster": cluster,
                "device": device,
                "passed": ok,
                "elapsed_s": round(elapsed, 2),
                "error": err,
                "verified_current": is_verified(env, ckpt, cluster),
            }
        )

        if not json_out:
            verdict = "PASS" if ok else "FAIL"
            line = f"{env_name}/{ckpt_name:<24} [{verdict}]  {device}  {elapsed:5.1f}s"
            if not ok:
                line += f"  {err}"
            print(line)

    # Custom-weights legs (#200): re-load each base's cached weights file via
    # the weights= path and require agreement with the canonical run.
    custom_outcomes: list[tuple[str, str, bool, str | None]] = []
    for leg in custom_legs:
        key = (leg.env_name, leg.base)
        if key not in passed_keys:
            custom_skips.append(
                (leg.env_name, leg.custom_id, f"baseline {leg.base} failed its own smoke-test")
            )
            continue
        weights_rel = _dominant_weights_file(base_captures.get(key))
        if weights_rel is None:
            custom_skips.append(
                (
                    leg.env_name,
                    leg.custom_id,
                    f"no single dominant weights file in {leg.base}'s capture",
                )
            )
            continue
        weights_path = Path(cache_root) / weights_rel
        if not weights_path.is_file():
            custom_skips.append(
                (leg.env_name, leg.custom_id, f"recorded weights file missing: {weights_path}")
            )
            continue

        start = time.monotonic()
        run_results = {}
        ok, err = verify_checkpoint(
            root=root,
            env_name=leg.env_name,
            checkpoint=leg.custom_id,
            device=device,
            setup_kwargs={},
            cache_root=cache_root,
            checkpoint_path=str(weights_path),
            results=run_results,
            timeout=verify_timeout,
        )
        if ok:
            mismatch = results_mismatch(baselines[key], run_results)
            if mismatch is not None:
                ok, err = False, f"weights= run diverges from {leg.base}: {mismatch}"
        elapsed = time.monotonic() - start
        custom_outcomes.append((leg.env_name, leg.custom_id, ok, err))

        env = manifest.environments[leg.env_name]
        ckpt = env.checkpoints.setdefault(leg.custom_id, CheckpointInfo())
        if ok:
            ckpt.verifications[cluster] = VerificationRecord(
                verified_at=now_iso(), verified_device=device
            )
            n_passed += 1
        else:
            ckpt.verifications[cluster] = VerificationRecord(last_error=f"smoke-test: {err}")
            n_failed += 1

        results.append(
            {
                "env": leg.env_name,
                "checkpoint": leg.custom_id,
                "base_checkpoint": leg.base,
                "cluster": cluster,
                "device": device,
                "passed": ok,
                "elapsed_s": round(elapsed, 2),
                "error": err,
                "verified_current": is_verified(env, ckpt, cluster),
            }
        )

        if not json_out:
            verdict = "PASS" if ok else "FAIL"
            line = (
                f"{leg.env_name}/{leg.custom_id:<24} [{verdict}]  {device}  {elapsed:5.1f}s"
                f"  (weights from {leg.base})"
            )
            if not ok:
                line += f"  {err}"
            print(line)

    if not json_out:
        for env_name, custom_id, reason in custom_skips:
            print(f"{env_name}/{custom_id:<24} [SKIP]  {reason}")

    with manifest_lock(root):
        fresh = load_manifest(root)
        if fresh is not None:
            for env_name, ckpt_name, ok, err, weight_files in outcomes:
                env_record = fresh.environments.get(env_name)
                ckpt_record = env_record.checkpoints.get(ckpt_name) if env_record else None
                if ckpt_record is None:
                    continue  # env/checkpoint removed while we were testing
                if ok:
                    ckpt_record.verifications[cluster] = VerificationRecord(
                        verified_at=now_iso(), verified_device=device
                    )
                else:
                    ckpt_record.verifications[cluster] = VerificationRecord(
                        last_error=f"smoke-test: {err}"
                    )
                apply_weights_record(
                    ckpt_record,
                    weight_files,
                    label=f"{env_name}/{ckpt_name}",
                    progress=None if json_out else print,
                )
            for env_name, custom_id, ok, err in custom_outcomes:
                env_record = fresh.environments.get(env_name)
                if env_record is None:
                    continue  # env removed while we were testing
                # Created on first pass — ':custom' entries are never fetched,
                # so nothing else ever writes them into the manifest.
                ckpt_record = env_record.checkpoints.setdefault(custom_id, CheckpointInfo())
                if ok:
                    ckpt_record.verifications[cluster] = VerificationRecord(
                        verified_at=now_iso(), verified_device=device
                    )
                else:
                    ckpt_record.verifications[cluster] = VerificationRecord(
                        last_error=f"smoke-test: {err}"
                    )
            save_manifest(fresh, root)
    update_and_push_manifest(root, quiet=True, push=not no_push)

    total_elapsed = time.monotonic() - total_start

    if json_out:
        print(
            json.dumps(
                {
                    "cluster": cluster,
                    "results": results,
                    "passed": n_passed,
                    "failed": n_failed,
                    "skipped": [
                        {"env": e, "checkpoint": c, "reason": r} for e, c, r in custom_skips
                    ],
                    "elapsed_s": round(total_elapsed, 2),
                },
                indent=2,
            )
        )
    else:
        summary = f"\n{n_passed} passed, {n_failed} failed"
        if custom_skips:
            summary += f", {len(custom_skips)} skipped"
        print(f"{summary} in {total_elapsed:.1f}s")

    return 0 if n_failed == 0 else 1
