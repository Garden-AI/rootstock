"""Tests for the ``rootstock sync`` CLI adapter.

Planner and executor are mocked (they have their own tests under
tests/commands/); what's under test is the adapter contract: flag handling,
exit codes, --dry-run/--json behavior, and the shared-install umask.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rootstock.batch import BuildItem, ItemResult, SyncPlan, SyncReport
from rootstock.commands.sync import cmd_sync


def _make_args(root: Path, **overrides):
    class _Args:
        pass

    args = _Args()
    args.source_dir = overrides.get("source_dir")
    args.root = str(root)
    args.cluster = overrides.get("cluster")
    args.env = overrides.get("env")
    args.checkpoint = overrides.get("checkpoint")
    args.rebuild = overrides.get("rebuild", False)
    args.upgrade = overrides.get("upgrade", False)
    args.phases = overrides.get("phases", "build,download,verify")
    args.jobs = overrides.get("jobs", 4)
    args.verify_jobs = overrides.get("verify_jobs", 1)
    args.device = overrides.get("device", "cuda")
    args.dry_run = overrides.get("dry_run", False)
    args.json = overrides.get("json", False)
    args.fail_fast = overrides.get("fail_fast", False)
    args.no_push = overrides.get("no_push", True)
    args.no_perm_check = overrides.get("no_perm_check", True)
    return args


@pytest.fixture
def stubbed(monkeypatch) -> dict:
    """Stub the planner/executor; tests set the plan/report to return."""
    state = {
        "plan": SyncPlan(),
        "report": SyncReport(),
        "plan_calls": [],
        "exec_calls": [],
    }

    def fake_plan(root, **kwargs):
        state["plan_calls"].append((root, kwargs))
        return state["plan"]

    def fake_execute(root, plan, **kwargs):
        state["exec_calls"].append((root, kwargs))
        return state["report"]

    monkeypatch.setattr("rootstock.commands.sync.plan_sync", fake_plan)
    monkeypatch.setattr("rootstock.commands.sync.execute_sync", fake_execute)
    return state


ONE_BUILD = SyncPlan(builds=[BuildItem("mace", "mace", "not built")])


def test_dry_run_plans_but_never_executes(tmp_path, stubbed):
    stubbed["plan"] = ONE_BUILD

    rc = cmd_sync(_make_args(tmp_path, dry_run=True))

    assert rc == 0
    assert len(stubbed["plan_calls"]) == 1
    assert stubbed["exec_calls"] == []
    # --dry-run must not stamp the layout marker.
    assert not (tmp_path / "layout.json").exists()


def test_empty_plan_short_circuits(tmp_path, stubbed, capsys):
    rc = cmd_sync(_make_args(tmp_path))

    assert rc == 0
    assert stubbed["exec_calls"] == []
    assert "Nothing to do" in capsys.readouterr().out


def test_execution_failure_exits_1(tmp_path, stubbed):
    stubbed["plan"] = ONE_BUILD
    stubbed["report"] = SyncReport(results=[ItemResult("build", "mace", None, "failed", "boom")])

    rc = cmd_sync(_make_args(tmp_path))

    assert rc == 1
    assert len(stubbed["exec_calls"]) == 1
    assert (tmp_path / "layout.json").exists(), "a mutating run stamps the layout marker"


def test_success_exits_0_and_forwards_knobs(tmp_path, stubbed):
    stubbed["plan"] = ONE_BUILD
    stubbed["report"] = SyncReport(results=[ItemResult("build", "mace", None, "ok", "not built")])

    rc = cmd_sync(
        _make_args(tmp_path, jobs=8, verify_jobs=2, device="cuda", fail_fast=True, upgrade=True)
    )

    assert rc == 0
    ((_, kwargs),) = stubbed["exec_calls"]
    assert kwargs["jobs"] == 8
    assert kwargs["verify_jobs"] == 2
    assert kwargs["fail_fast"] is True
    assert kwargs["upgrade"] is True
    assert kwargs["push"] is False  # from no_push=True


def test_phase_spec_is_validated_and_canonicalized(tmp_path, stubbed):
    assert cmd_sync(_make_args(tmp_path, phases="verify,nonsense")) == 2
    assert stubbed["plan_calls"] == []

    assert cmd_sync(_make_args(tmp_path, phases="verify,build")) == 0
    ((_, kwargs),) = stubbed["plan_calls"]
    assert kwargs["phases"] == ("build", "verify"), "canonical order, however spelled"


def test_missing_source_dir_is_a_usage_error(tmp_path, stubbed):
    rc = cmd_sync(_make_args(tmp_path, source_dir=str(tmp_path / "nope")))
    assert rc == 2
    assert stubbed["plan_calls"] == []


def test_json_dry_run_emits_the_plan_on_stdout(tmp_path, stubbed, capsys):
    stubbed["plan"] = ONE_BUILD

    rc = cmd_sync(_make_args(tmp_path, dry_run=True, json=True))

    assert rc == 0
    document = json.loads(capsys.readouterr().out)
    assert document["plan"]["builds"] == [
        {"env_name": "mace", "source": "mace", "reason": "not built"}
    ]


def test_json_run_emits_results_and_keeps_stdout_clean(tmp_path, stubbed, capsys):
    stubbed["plan"] = ONE_BUILD
    stubbed["report"] = SyncReport(results=[ItemResult("build", "mace", None, "ok", "not built")])

    rc = cmd_sync(_make_args(tmp_path, json=True))

    assert rc == 0
    out = capsys.readouterr().out
    document = json.loads(out)  # stdout is exactly one JSON document
    assert document["counts"] == {"ok": 1, "failed": 0, "skipped": 0}
    assert document["results"][0]["env_name"] == "mace"


def test_sync_overrides_restrictive_umask(tmp_path, stubbed):
    """Everything sync writes to a shared install must be group-writable,
    whatever the maintainer's personal umask says."""
    old = os.umask(0o077)
    try:
        assert cmd_sync(_make_args(tmp_path, dry_run=True)) == 0
        assert os.umask(0o022) == 0o002
    finally:
        os.umask(old)


def test_planner_errors_are_reported_cleanly(tmp_path, monkeypatch, capsys):
    from rootstock.operations import OperationError

    def exploding_plan(root, **kwargs):
        raise OperationError("Unknown environment(s): nope")

    monkeypatch.setattr("rootstock.commands.sync.plan_sync", exploding_plan)

    rc = cmd_sync(_make_args(tmp_path, env=["nope"]))

    assert rc == 1
    assert "Unknown environment" in capsys.readouterr().err
