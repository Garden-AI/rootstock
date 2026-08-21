"""``rootstock smoke-test`` — re-verify checkpoints already in the manifest.

Selection is checkpoint-first (#208): each canonical id declared by a built
env serving this cluster is resolved with the same cluster-aware resolution
the calculator uses, and tested in the env it resolves to — so a
cluster-specific variant shadows the universal env per id, exactly as users
experience it.

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
    CheckpointNotFoundError,
    find_env_for_checkpoint,
    is_custom_checkpoint,
    parse_checkpoints_dict,
    parse_clusters_list,
    parse_custom_checkpoint_ids,
    resolve_checkpoint,
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


def _declared_ids(root: Path, env_name: str) -> list[str] | None:
    """Canonical ids the env's built source declares, in declaration order —
    or ``None`` when there is no source on disk to parse (a pre-source-copy
    build), telling the caller to fall back to the env's manifest records.
    A malformed source contributes nothing, matching
    ``list_declared_checkpoints``."""
    source = root / "envs" / env_name / "env_source.py"
    if not source.exists():
        return None
    try:
        return list(parse_checkpoints_dict(source))
    except ValueError:
        return []


def _fetched_record(manifest: Manifest, ckpt_id: str) -> CheckpointInfo | None:
    """Any env's record showing ``ckpt_id`` was fetched. The weights cache is
    shared and keyed by checkpoint, not env — one env's download serves them
    all — so fetched anywhere is fetched."""
    for env in manifest.environments.values():
        record = env.checkpoints.get(ckpt_id)
        if record is not None and record.fetched_at:
            return record
    return None


def _resolves_to(root: Path, ckpt_id: str, cluster: str, env_name: str) -> bool:
    """Whether ``ckpt_id`` resolves to ``env_name`` on ``cluster``."""
    try:
        return find_env_for_checkpoint(root, ckpt_id, cluster)[0] == env_name
    except CheckpointNotFoundError:
        return False


def _select(
    root: Path, manifest, env_filter: str | None, checkpoint_filter: str | None, cluster: str
) -> tuple[list[tuple[str, str, EnvironmentInfo, CheckpointInfo]], list[tuple[str, str, str]]]:
    """Pick which (env, checkpoint) pairs to test — checkpoint-first (#208).

    The unit of testing is the canonical id, not the env record: collect the
    ids declared by built envs serving this cluster, resolve each one exactly
    like the calculator would (a cluster-specific variant beats the universal
    env), and test it in the env it resolves to — also the env whose record
    the outcome lands in. Per-id shadowing falls out: the variant's override
    is what gets tested on its cluster, while ids the variant doesn't declare
    keep being tested via the universal env. ``env_filter`` therefore means
    "ids that resolve to that env on this cluster".

    An id counts as fetched when *any* env's record says so (the cache is
    shared, keyed by checkpoint); the resolved env's record inherits the
    donor's ``fetched_at`` so it never reads "verified but never fetched".
    Ids fetched nowhere are skipped — smoke-test never downloads.

    Envs built before sources were copied into the env dir have nothing to
    parse; their manifest records are tested in place (the old env-first
    walk).

    Returns the selection plus ``(env, checkpoint, reason)`` notes for ids
    that fail to resolve (e.g. declared by two same-specificity envs) — a
    nightly run reports those and keeps testing everything else.
    """
    selected: list[tuple[str, str, EnvironmentInfo, CheckpointInfo]] = []
    skipped: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    sourceless: list[tuple[str, EnvironmentInfo]] = []

    for env_name, env in manifest.environments.items():
        if not _serves_here(root, env_name, env, cluster):
            # A cluster-restricted variant can only be verified by a machine
            # it serves (#208); its own cluster's chain covers it.
            continue
        declared = _declared_ids(root, env_name)
        if declared is None:
            sourceless.append((env_name, env))
            continue
        for ckpt_id in declared:
            if ckpt_id in seen:
                continue
            seen.add(ckpt_id)
            if checkpoint_filter is not None and ckpt_id != checkpoint_filter:
                continue
            try:
                resolved, _ = find_env_for_checkpoint(root, ckpt_id, cluster)
            except CheckpointNotFoundError as exc:
                if env_filter is None or env_name == env_filter:
                    skipped.append((env_name, ckpt_id, str(exc)))
                continue
            if env_filter is not None and resolved != env_filter:
                continue
            host = manifest.environments.get(resolved)
            if host is None:
                skipped.append((resolved, ckpt_id, "resolved env is not in the manifest"))
                continue
            donor = _fetched_record(manifest, ckpt_id)
            if donor is None:
                # Smoke-test never downloads. Skip checkpoints that have never
                # been fetched by any env.
                continue
            record = host.checkpoints.setdefault(ckpt_id, CheckpointInfo())
            if record.fetched_at is None:
                record.fetched_at = donor.fetched_at
            selected.append((resolved, ckpt_id, host, record))

    for env_name, env in sourceless:
        if env_filter is not None and env_name != env_filter:
            continue
        for ckpt_name, ckpt in env.checkpoints.items():
            if ckpt_name in seen:
                continue  # already tested via the env it resolves to
            if checkpoint_filter is not None and ckpt_name != checkpoint_filter:
                continue
            if is_custom_checkpoint(ckpt_name):
                # ':custom' records track the weights= leg below; they carry
                # no shipped weights for the canonical loop to verify.
                continue
            if ckpt.fetched_at is None:
                continue
            selected.append((env_name, ckpt_name, env, ckpt))

    return selected, skipped


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
    """Plan one weights= leg per ``<family>:custom`` id declared by a built
    source serving this cluster (#200).

    Like the canonical selection, planning is checkpoint-first (#208): each
    custom id is resolved for this cluster, and the leg runs in — and is
    recorded on — the env it resolves to. The base is the first canonical id
    (in the resolved env's CHECKPOINTS declaration order — env authors lead
    with the family's plainest-loading checkpoint) of the same family that
    is fetched anywhere and resolves to the same env, so the canonical loop
    is guaranteed to have produced its baseline. Returns the legs plus
    ``(env, custom_id, reason)`` for entries that can't run this time.
    """
    legs: list[CustomLeg] = []
    skipped: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for env_name, env in manifest.environments.items():
        if not _serves_here(root, env_name, env, cluster):
            continue
        source = Path(root) / "envs" / env_name / "env_source.py"
        if not source.exists():
            continue
        try:
            custom_ids = parse_custom_checkpoint_ids(source)
        except ValueError:
            continue
        for custom_id in custom_ids:
            if custom_id in seen:
                continue
            seen.add(custom_id)
            if ckpt_filter is not None and ckpt_filter != custom_id:
                continue
            try:
                resolved = resolve_checkpoint(root, custom_id, cluster).env_name
            except CheckpointNotFoundError as exc:
                if env_filter is None or env_name == env_filter:
                    skipped.append((env_name, custom_id, str(exc)))
                continue
            if env_filter is not None and resolved != env_filter:
                continue
            if resolved not in manifest.environments:
                skipped.append((resolved, custom_id, "resolved env is not in the manifest"))
                continue
            host_source = Path(root) / "envs" / resolved / "env_source.py"
            try:
                declared = parse_checkpoints_dict(host_source)
                families = [
                    c.removesuffix(CUSTOM_CHECKPOINT_SUFFIX)
                    for c in parse_custom_checkpoint_ids(host_source)
                ]
            except (OSError, ValueError):
                continue
            family = custom_id.removesuffix(CUSTOM_CHECKPOINT_SUFFIX)
            base = next(
                (
                    c
                    for c in declared
                    if _family_of(c, families) == family
                    and _fetched_record(manifest, c) is not None
                    and _resolves_to(root, c, cluster, resolved)
                ),
                None,
            )
            if base is None:
                skipped.append(
                    (
                        resolved,
                        custom_id,
                        f"no fetched '{family}' checkpoint to borrow weights from",
                    )
                )
                continue
            legs.append(CustomLeg(resolved, custom_id, base))
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

    selected, resolve_skips = _select(root, manifest, env_filter, ckpt_filter, cluster)
    if ckpt_filter is not None and is_custom_checkpoint(ckpt_filter):
        # A ':custom' filter matches no canonical row; run just the base
        # checkpoint(s) the leg needs for its comparison.
        needed = {(leg.env_name, leg.base) for leg in custom_legs}
        selected = [
            t
            for t in _select(root, manifest, env_filter, None, cluster)[0]
            if (t[0], t[1]) in needed
        ]
    custom_skips = resolve_skips + custom_skips

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
    outcomes: list[tuple[str, str, bool, str | None, list[dict] | None, str | None]] = []

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
                setup_kwargs={},  # empty → verify falls back to the env's VERIFY_KWARGS
                cache_root=cache_root,
                weights_capture_path=str(capture_path),
                results=run_results,
                timeout=verify_timeout,
            )
            weight_files = read_weights_capture(capture_path)
        elapsed = time.monotonic() - start
        outcomes.append((env_name, ckpt_name, ok, err, weight_files, ckpt.fetched_at))

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
            setup_kwargs={},  # empty → VERIFY_KWARGS fallback (keyed by the :custom id)
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
            for env_name, ckpt_name, ok, err, weight_files, fetched_at in outcomes:
                env_record = fresh.environments.get(env_name)
                if env_record is None:
                    continue  # env removed while we were testing
                # Created on first pass when the resolved env has no record of
                # an id it now hosts (a variant tested checkpoint-first for
                # the first time, #208); the donor's fetch stamp rides along
                # so the record never reads "verified but never fetched".
                ckpt_record = env_record.checkpoints.setdefault(ckpt_name, CheckpointInfo())
                if ckpt_record.fetched_at is None:
                    ckpt_record.fetched_at = fetched_at
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
