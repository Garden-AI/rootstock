"""``rootstock usage`` — report on, compact, and push the usage-record spool.

The spool ({cache_root}/usage/) fills with one small JSON record per
calculator session (see rootstock/usage.py). ``report`` aggregates it
read-only; ``compact`` folds raw records into per-month rollup files so the
spool doesn't accumulate thousands of tiny files; ``push`` sends the
aggregated rollup rows to the dashboard backend. All are login-node,
maintainer-side operations — the write side never needs them.
"""

from __future__ import annotations

import json
import sys

from ..client import RootstockClient
from ..config import load_config
from ..manifest import load_manifest
from ..usage import SpoolSummary, compact_spool, summarize_spool, usage_dir
from .common import get_root_or_exit, resolve_cache_root


def _print_rows(summary: SpoolSummary) -> None:
    if not summary.rows:
        print("No usage recorded yet.")
        return

    headers = (
        "month",
        "cluster",
        "env",
        "checkpoint",
        "device",
        "client",
        "sessions",
        "calls",
        "hours",
        "users",
    )
    table = [
        (
            row["month"],
            row["cluster"] or "-",
            row["env"],
            row["checkpoint"],
            row["device"],
            row["client"] or "-",
            str(row["sessions"]),
            str(row["n_calculations"]),
            f"{row['duration_s'] / 3600:.1f}",
            str(row["unique_users"]),
        )
        for row in summary.rows
    ]
    widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(headers)]
    for line in (headers, *table):
        print("  ".join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip())


def cmd_usage_report(args) -> int:
    """Aggregate the spool (rollups + raw records), read-only."""
    root = get_root_or_exit(args)
    cache_root = resolve_cache_root(root, args.cache_root)

    summary = summarize_spool(cache_root)
    if summary is None:
        print(
            f"No usage spool at {usage_dir(cache_root)} — usage collection is "
            "off for this install (rootstock setup-perms provisions it).",
            file=sys.stderr,
        )
        return 1

    if args.json:
        payload = {
            "rows": summary.rows,
            "unique_users": summary.unique_users,
            "skipped": summary.skipped,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Usage spool: {usage_dir(cache_root)}")
    _print_rows(summary)
    if summary.rows:
        print(f"Unique users overall: {summary.unique_users}")
    if summary.skipped:
        print(f"({summary.skipped} unreadable file(s) skipped)", file=sys.stderr)
    return 0


def cmd_usage_push(args) -> int:
    """Push the aggregated rollup rows to the dashboard backend.

    Pushes the same aggregation ``report`` shows — rollup files plus any
    not-yet-compacted raw records — filed under the manifest's cluster name.
    The backend stores rollups per month and replaces only the months
    present in the push, so pushing repeatedly (e.g. from the smoke-test
    cron) is idempotent and never erases previously pushed history. Only
    derived counts are sent; the salted user hashes stay in the spool.
    """
    root = get_root_or_exit(args)
    cache_root = resolve_cache_root(root, args.cache_root)
    config = load_config()

    valid, error = config.validate()
    if not valid:
        print(f"Error: {error}", file=sys.stderr)
        print(
            "Configure API credentials in ~/.config/rootstock/config.toml",
            file=sys.stderr,
        )
        return 1
    url = config.resolve_usage_api_url()
    if not url:
        print(
            "Error: no usage endpoint — api_url doesn't follow the standard "
            "rootstock-admin naming, so set usage_api_url in "
            "~/.config/rootstock/config.toml explicitly.",
            file=sys.stderr,
        )
        return 1

    manifest = load_manifest(root)
    if manifest is None or not manifest.clusters:
        print(
            f"No manifest at {root}/manifest.json — pushes are filed under "
            "cluster names; run 'rootstock manifest init --cluster <name>' "
            "first.",
            file=sys.stderr,
        )
        return 1

    summary = summarize_spool(cache_root)
    if summary is None:
        print(
            f"No usage spool at {usage_dir(cache_root)} — usage collection is "
            "off for this install (rootstock setup-perms provisions it).",
            file=sys.stderr,
        )
        return 1
    if not summary.rows:
        print("No usage recorded yet — nothing to push.")
        return 0

    # Rows carry the cluster their sessions ran on — on a shared install
    # (sophia/polaris) one spool holds several machines' usage, so each
    # cluster gets its own push (#208). Rows predating the per-session stamp
    # (or from root=-only sessions) fall back to the install's home cluster.
    home = manifest.clusters[0]
    by_cluster: dict[str, list[dict]] = {}
    for row in summary.rows:
        by_cluster.setdefault(row.get("cluster") or home, []).append(row)

    if args.dry_run:
        for cluster, rows in sorted(by_cluster.items()):
            payload = {"cluster": cluster, "rows": rows}
            print(f"Would POST to {url}:")
            print(json.dumps(payload, indent=2))
        return 0

    client = RootstockClient(config)
    all_ok = True
    for cluster, rows in sorted(by_cluster.items()):
        success, message = client.push_usage(cluster, rows)
        if success:
            print(message)
        else:
            print(f"Error ({cluster}): {message}", file=sys.stderr)
            all_ok = False
    return 0 if all_ok else 1


def cmd_usage_compact(args) -> int:
    """Fold raw records into per-month rollup files."""
    root = get_root_or_exit(args)
    cache_root = resolve_cache_root(root, args.cache_root)

    summary = compact_spool(cache_root)
    if summary is None:
        print(
            f"No usage spool at {usage_dir(cache_root)} — nothing to compact.",
            file=sys.stderr,
        )
        return 1

    print(f"Compacted {summary.raw_files} record(s) into monthly rollups.")
    if summary.kept:
        # The spool is sticky: only a record's owner (or the spool's owner)
        # can delete it, and merging without deleting would double-count.
        print(
            f"{summary.kept} record(s) left in place — owned by other users; "
            "the spool owner's compact run will pick them up.",
            file=sys.stderr,
        )
    if summary.skipped:
        print(f"({summary.skipped} unreadable file(s) skipped)", file=sys.stderr)
    return 0
