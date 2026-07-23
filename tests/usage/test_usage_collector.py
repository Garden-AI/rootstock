"""Collector semantics: aggregate the spool, compact without double-counting.

compact's core invariant is delete-then-merge: a raw record the caller can't
delete (the spool is sticky — only the record's owner or the spool's owner
can unlink it) is left alone rather than merged, because merging without
deleting would count it again on the next run.
"""

from __future__ import annotations

import json
import os

import pytest

from rootstock.usage import (
    ROLLUP_PREFIX,
    compact_spool,
    record_session,
    summarize_spool,
    usage_dir,
)


def _spool(tmp_path):
    spool = usage_dir(tmp_path)
    spool.mkdir()
    return spool


def _write(
    cache_root,
    month="2026-07",
    checkpoint="mace-mp-0-medium",
    n=10,
    duration=60.0,
    client="calculator",
):
    path = record_session(
        root=cache_root,
        cache_root=cache_root,
        env_name="mace",
        checkpoint=checkpoint,
        is_local=False,
        device="cuda",
        client=client,
        started_at=f"{month}-23T01:02:03+00:00",
        duration_s=duration,
        n_calculations=n,
    )
    assert path is not None
    return path


def test_summarize_aggregates_by_month_and_checkpoint(tmp_path):
    _spool(tmp_path)
    _write(tmp_path, n=10)
    _write(tmp_path, n=5)
    _write(tmp_path, checkpoint="uma-s-1p1", n=2)
    _write(tmp_path, month="2026-06", n=1)

    summary = summarize_spool(tmp_path)
    assert summary.raw_files == 4
    assert summary.skipped == 0
    rows = [(r["month"], r["checkpoint"], r["sessions"], r["n_calculations"]) for r in summary.rows]
    assert rows == [
        ("2026-06", "mace-mp-0-medium", 1, 1),
        ("2026-07", "mace-mp-0-medium", 2, 15),
        ("2026-07", "uma-s-1p1", 1, 2),
    ]


def test_serve_rows_stay_distinct_and_null_calls_sum_as_zero(tmp_path):
    """client is a rollup key — serve/LAMMPS adoption must stay visible after
    compaction — and serve's n_calculations=null (the parent process never
    sees the i-PI traffic) counts the session while summing nothing."""
    _spool(tmp_path)
    _write(tmp_path, n=10)
    _write(tmp_path, client="serve", n=None)
    compact_spool(tmp_path)
    _write(tmp_path, client="serve", n=None)  # fresh record after compaction

    summary = summarize_spool(tmp_path)
    rows = [(r["client"], r["sessions"], r["n_calculations"]) for r in summary.rows]
    assert rows == [
        ("calculator", 1, 10),
        ("serve", 2, 0),
    ]


def test_summarize_missing_spool_returns_none(tmp_path):
    assert summarize_spool(tmp_path) is None
    assert compact_spool(tmp_path) is None


def test_summarize_skips_garbage_without_dying(tmp_path):
    spool = _spool(tmp_path)
    _write(tmp_path)
    (spool / "not-json.json").write_text("{{{")
    (spool / "future.json").write_text(json.dumps({"schema_version": 99}))

    summary = summarize_spool(tmp_path)
    assert summary.raw_files == 1
    assert summary.skipped == 2
    assert summary.rows[0]["sessions"] == 1


def test_compact_rolls_up_and_removes_raw_records(tmp_path):
    spool = _spool(tmp_path)
    _write(tmp_path, n=10, duration=30.0)
    _write(tmp_path, n=5, duration=30.0)
    _write(tmp_path, month="2026-06", n=1)

    result = compact_spool(tmp_path)
    assert result.raw_files == 3
    assert result.kept == 0

    names = sorted(p.name for p in spool.glob("*.json"))
    assert names == [f"{ROLLUP_PREFIX}2026-06.json", f"{ROLLUP_PREFIX}2026-07.json"]

    july = json.loads((spool / f"{ROLLUP_PREFIX}2026-07.json").read_text())
    (row,) = july["rows"]
    assert (row["sessions"], row["n_calculations"], row["duration_s"]) == (2, 15, 60.0)


def test_compact_is_idempotent_and_merges_new_records(tmp_path):
    _spool(tmp_path)
    _write(tmp_path, n=10)
    compact_spool(tmp_path)
    assert compact_spool(tmp_path).raw_files == 0  # nothing new: a no-op

    _write(tmp_path, n=5)
    compact_spool(tmp_path)

    summary = summarize_spool(tmp_path)
    (row,) = summary.rows
    assert (row["sessions"], row["n_calculations"]) == (2, 15)


def test_report_sees_rollups_plus_fresh_raw_records(tmp_path):
    _spool(tmp_path)
    _write(tmp_path, n=10)
    compact_spool(tmp_path)
    _write(tmp_path, n=7)  # arrived after the last compact

    (row,) = summarize_spool(tmp_path).rows
    assert (row["sessions"], row["n_calculations"]) == (2, 17)


def test_unique_users_survive_compaction_without_double_count(tmp_path, monkeypatch):
    """Distinct-user counting is a set union: the same hash arriving via a
    rollup and via a fresh raw record must count once, and per-row hashes
    never appear in summaries — only counts do."""
    spool = _spool(tmp_path)
    _write(tmp_path)  # the real user
    monkeypatch.setattr("rootstock.usage.getpass.getuser", lambda: "user-b")
    _write(tmp_path)
    compact_spool(tmp_path)
    _write(tmp_path)  # user-b again, arriving after the compaction

    summary = summarize_spool(tmp_path)
    (row,) = summary.rows
    assert row["sessions"] == 3
    assert row["unique_users"] == 2
    assert summary.unique_users == 2
    assert "users" not in row  # hashes stay in the spool, not in summaries

    compact_spool(tmp_path)
    rollup = json.loads((spool / f"{ROLLUP_PREFIX}2026-07.json").read_text())
    assert len(rollup["rows"][0]["users"]) == 2  # exact union persisted


def test_records_without_user_hash_still_count_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("rootstock.usage._user_hash", lambda spool: None)
    _spool(tmp_path)
    _write(tmp_path)

    summary = summarize_spool(tmp_path)
    (row,) = summary.rows
    assert row["sessions"] == 1
    assert row["unique_users"] == 0


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores directory modes")
def test_compact_keeps_undeletable_records_unmerged(tmp_path):
    """Delete-then-merge: what can't be deleted must not be merged, or the
    next compact run would double-count it."""
    spool = _spool(tmp_path)
    _write(tmp_path, n=10)
    spool.chmod(0o555)  # unlink now fails, like another user's file in a sticky dir
    try:
        result = compact_spool(tmp_path)
    finally:
        spool.chmod(0o755)

    assert result.kept == 1
    assert result.raw_files == 0
    # The record is still there, uncounted by any rollup, for the owner's run.
    (row,) = summarize_spool(tmp_path).rows
    assert row["sessions"] == 1
    assert not any(p.name.startswith(ROLLUP_PREFIX) for p in spool.iterdir())
