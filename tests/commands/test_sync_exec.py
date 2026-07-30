"""The ``rootstock sync`` executor.

Operations are mocked; what's under test is the orchestration contract:
phase ordering, dependent-skip on failure, keep-going vs --fail-fast, the
verify device pool, and the single trailing manifest refresh.
"""

from __future__ import annotations

import threading

import pytest

from rootstock.batch import BuildItem, CheckpointItem, SyncPlan, execute_sync
from rootstock.operations import OperationError


@pytest.fixture
def calls(monkeypatch) -> dict:
    """Mock the three operations + the final refresh, recording every call."""
    recorded: dict = {"build": [], "download": [], "verify": [], "refresh": []}
    lock = threading.Lock()

    def fake_install(root, source, **kwargs):
        with lock:
            recorded["build"].append((source, kwargs))
        if "boom" in str(source):
            raise OperationError(f"build of {source} exploded")

    def fake_fetch(root, checkpoint, **kwargs):
        with lock:
            recorded["download"].append((checkpoint, kwargs))
        if "unfetchable" in checkpoint:
            raise OperationError("download failed: hub unreachable")

    def fake_verify(root, checkpoint, **kwargs):
        with lock:
            recorded["verify"].append((checkpoint, kwargs))
        if "unverifiable" in checkpoint:
            raise OperationError("verify failed: CUDA OOM")

    def fake_refresh(root, **kwargs):
        with lock:
            recorded["refresh"].append(kwargs)
        return True

    monkeypatch.setattr("rootstock.operations.install_environment", fake_install)
    monkeypatch.setattr("rootstock.operations.fetch_checkpoint", fake_fetch)
    monkeypatch.setattr("rootstock.operations.verify_fetched_checkpoint", fake_verify)
    monkeypatch.setattr("rootstock.operations.update_and_push_manifest", fake_refresh)
    return recorded


def plan_for(*, builds=(), downloads=(), verifies=()) -> SyncPlan:
    return SyncPlan(
        builds=[BuildItem(env, env, "test") for env in builds],
        downloads=[CheckpointItem(env, ckpt, "test") for env, ckpt in downloads],
        verifies=[CheckpointItem(env, ckpt, "test") for env, ckpt in verifies],
    )


def by_status(report, status):
    return {(r.phase, r.label) for r in report.results if r.status == status}


def test_happy_path_runs_all_phases_and_refreshes_once(tmp_path, calls):
    plan = plan_for(
        builds=("mace", "uma"),
        downloads=(("mace", "m1"), ("uma", "u1")),
        verifies=(("mace", "m1"), ("uma", "u1")),
    )

    report = execute_sync(tmp_path, plan, say=lambda _: None)

    assert report.counts() == {"ok": 6, "failed": 0, "skipped": 0}
    assert len(calls["refresh"]) == 1, "one manifest refresh, at the end"
    # Operations must never refresh or push per-item.
    assert all(kw.get("push") is False for _, kw in calls["build"])
    assert all(kw.get("refresh") is False for _, kw in calls["download"])
    assert all(kw.get("refresh") is False for _, kw in calls["verify"])


def test_failed_build_skips_its_checkpoints_but_not_others(tmp_path, calls):
    plan = plan_for(
        builds=("boom", "uma"),
        downloads=(("boom", "b1"), ("uma", "u1")),
        verifies=(("boom", "b1"), ("uma", "u1")),
    )

    report = execute_sync(tmp_path, plan, say=lambda _: None)

    assert by_status(report, "failed") == {("build", "boom")}
    assert by_status(report, "skipped") == {("download", "boom/b1"), ("verify", "boom/b1")}
    skipped = [r for r in report.results if r.status == "skipped"]
    assert all("did not build" in r.reason for r in skipped)
    # uma's items all ran, and the batch still refreshed (some work succeeded).
    assert by_status(report, "ok") == {
        ("build", "uma"),
        ("download", "uma/u1"),
        ("verify", "uma/u1"),
    }
    assert len(calls["refresh"]) == 1


def test_failed_download_skips_only_that_verify(tmp_path, calls):
    plan = plan_for(
        downloads=(("mace", "unfetchable"), ("mace", "m2")),
        verifies=(("mace", "unfetchable"), ("mace", "m2")),
    )

    report = execute_sync(tmp_path, plan, say=lambda _: None)

    assert by_status(report, "failed") == {("download", "mace/unfetchable")}
    assert by_status(report, "skipped") == {("verify", "mace/unfetchable")}
    assert ("verify", "mace/m2") in by_status(report, "ok")


def test_nothing_succeeded_means_no_refresh(tmp_path, calls):
    plan = plan_for(builds=("boom",))

    report = execute_sync(tmp_path, plan, say=lambda _: None)

    assert report.counts()["failed"] == 1
    assert calls["refresh"] == [], "a batch where nothing improved must not push"


def test_fail_fast_aborts_later_phases(tmp_path, calls):
    plan = plan_for(
        builds=("boom",),
        downloads=(("uma", "u1"),),
        verifies=(("uma", "u1"),),
    )

    report = execute_sync(tmp_path, plan, fail_fast=True, say=lambda _: None)

    assert by_status(report, "failed") == {("build", "boom")}
    skipped = {(r.phase, r.label, r.reason) for r in report.results if r.status == "skipped"}
    assert skipped == {
        ("download", "uma/u1", "--fail-fast"),
        ("verify", "uma/u1", "--fail-fast"),
    }
    assert calls["download"] == [] and calls["verify"] == []


def test_verify_devices_round_robin_with_plain_cuda(tmp_path, calls):
    plan = plan_for(verifies=tuple(("mace", f"m{i}") for i in range(6)))

    execute_sync(tmp_path, plan, verify_jobs=2, device="cuda", say=lambda _: None)

    devices = {kw["device"] for _, kw in calls["verify"]}
    assert devices <= {"cuda:0", "cuda:1"}
    assert len(calls["verify"]) == 6


def test_explicit_device_is_never_rewritten(tmp_path, calls):
    plan = plan_for(verifies=(("mace", "m1"), ("mace", "m2")))

    execute_sync(tmp_path, plan, verify_jobs=2, device="cuda:3", say=lambda _: None)

    assert {kw["device"] for _, kw in calls["verify"]} == {"cuda:3"}


def test_unexpected_exceptions_are_kept_going_too(tmp_path, calls, monkeypatch):
    """A bug in one item (not an OperationError) still must not sink the batch."""

    def broken_fetch(root, checkpoint, **kwargs):
        raise ValueError("unexpected")

    monkeypatch.setattr("rootstock.operations.fetch_checkpoint", broken_fetch)
    plan = plan_for(downloads=(("mace", "m1"),), verifies=(("mace", "m1"),))

    report = execute_sync(tmp_path, plan, say=lambda _: None)

    assert by_status(report, "failed") == {("download", "mace/m1")}
    assert by_status(report, "skipped") == {("verify", "mace/m1")}


def test_failure_output_is_flushed_with_the_error(tmp_path, calls, monkeypatch):
    """Buffered progress lines surface when (and only when) an item fails."""

    def chatty_failing_install(root, source, progress=None, **kwargs):
        progress("step 1: resolving")
        progress("step 2: exploding")
        raise OperationError("no luck")

    monkeypatch.setattr("rootstock.operations.install_environment", chatty_failing_install)
    lines: list[str] = []

    execute_sync(tmp_path, plan_for(builds=("mace",)), say=lines.append)

    joined = "\n".join(lines)
    assert "step 2: exploding" in joined
    assert "no luck" in joined
