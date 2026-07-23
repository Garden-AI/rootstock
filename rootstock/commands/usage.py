"""``rootstock usage`` — report on and compact the usage-record spool.

The spool ({cache_root}/usage/) fills with one small JSON record per
calculator session (see rootstock/usage.py). ``report`` aggregates it
read-only; ``compact`` folds raw records into per-month rollup files so the
spool doesn't accumulate thousands of tiny files. Both are login-node,
maintainer-side operations — the write side never needs them.
"""

from __future__ import annotations

import json
import sys

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
