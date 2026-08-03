"""Tests for the ``rootstock prune`` CLI adapter.

Planner and executor are mocked (they have their own tests under
tests/commands/); what's under test is the adapter contract: the confirm
gate, exit codes, --dry-run/--json behavior, knob forwarding, and the
shared-install umask.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rootstock.batch import ItemResult, PruneEnvItem, PrunePlan, SyncReport
from rootstock.commands.prune import cmd_prune


def _make_args(root: Path, **overrides):
    args = SimpleNamespace()
    args.source_dir = overrides.get("source_dir")
    args.root = str(root)
    args.cluster = overrides.get("cluster")
    args.env = overrides.get("env")
    args.checkpoint = overrides.get("checkpoint")
    args.gc_only = overrides.get("gc_only", False)
    args.deep = overrides.get("deep", False)
    args.min_age = overrides.get("min_age", 24.0)
    args.yes = overrides.get("yes", True)
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
        "plan": PrunePlan(),
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

    monkeypatch.setattr("rootstock.commands.prune.plan_prune", fake_plan)
    monkeypatch.setattr("rootstock.commands.prune.execute_prune", fake_execute)
    return state


ONE_ENV = PrunePlan(envs=[PruneEnvItem("orb", "no registered source")])
ENV_OK = SyncReport(results=[ItemResult("env", "orb", None, "ok", "no registered source")])
ENV_FAILED = SyncReport(results=[ItemResult("env", "orb", None, "failed", "boom")])


def test_dry_run_plans_but_never_executes(tmp_path, stubbed):
    stubbed["plan"] = ONE_ENV

    rc = cmd_prune(_make_args(tmp_path, dry_run=True, yes=False))

    assert rc == 0
    assert len(stubbed["plan_calls"]) == 1
    assert stubbed["exec_calls"] == []
    # --dry-run must not stamp the layout marker.
    assert not (tmp_path / "layout.json").exists()


def test_empty_plan_short_circuits(tmp_path, stubbed, capsys):
    rc = cmd_prune(_make_args(tmp_path))

    assert rc == 0
    assert stubbed["exec_calls"] == []
    assert "Nothing to prune" in capsys.readouterr().out


def test_no_yes_without_a_tty_is_a_usage_error(tmp_path, stubbed, capsys):
    """A batch job that forgot --yes must fail loudly, not 'succeed' having
    deleted nothing. (pytest's captured stdin is not a tty.)"""
    stubbed["plan"] = ONE_ENV

    rc = cmd_prune(_make_args(tmp_path, yes=False))

    assert rc == 2
    assert stubbed["exec_calls"] == []
    assert "refusing to delete without --yes" in capsys.readouterr().err
    assert not (tmp_path / "layout.json").exists()


def test_interactive_decline_exits_0_without_executing(tmp_path, stubbed, monkeypatch, capsys):
    stubbed["plan"] = ONE_ENV
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda: "n")

    rc = cmd_prune(_make_args(tmp_path, yes=False))

    assert rc == 0
    assert stubbed["exec_calls"] == []
    assert "Aborted" in capsys.readouterr().out


def test_interactive_confirm_executes(tmp_path, stubbed, monkeypatch):
    stubbed["plan"] = ONE_ENV
    stubbed["report"] = ENV_OK
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda: "y")

    rc = cmd_prune(_make_args(tmp_path, yes=False))

    assert rc == 0
    assert len(stubbed["exec_calls"]) == 1


def test_execution_failure_exits_1(tmp_path, stubbed):
    stubbed["plan"] = ONE_ENV
    stubbed["report"] = ENV_FAILED

    rc = cmd_prune(_make_args(tmp_path))

    assert rc == 1
    assert len(stubbed["exec_calls"]) == 1
    assert (tmp_path / "layout.json").exists(), "a mutating run stamps the layout marker"


def test_success_exits_0_and_forwards_knobs(tmp_path, stubbed):
    stubbed["plan"] = ONE_ENV
    stubbed["report"] = ENV_OK

    rc = cmd_prune(
        _make_args(
            tmp_path,
            env=["orb"],
            checkpoint=["orb-v2"],
            gc_only=True,
            deep=True,
            min_age=1.5,
            fail_fast=True,
        )
    )

    assert rc == 0
    ((_, plan_kwargs),) = stubbed["plan_calls"]
    assert plan_kwargs["envs"] == ["orb"]
    assert plan_kwargs["checkpoints"] == ["orb-v2"]
    assert plan_kwargs["gc_only"] is True
    assert plan_kwargs["deep"] is True
    assert plan_kwargs["min_age_hours"] == 1.5
    ((_, exec_kwargs),) = stubbed["exec_calls"]
    assert exec_kwargs["fail_fast"] is True
    assert exec_kwargs["push"] is False  # from no_push=True


def test_missing_source_dir_is_a_usage_error(tmp_path, stubbed):
    rc = cmd_prune(_make_args(tmp_path, source_dir=str(tmp_path / "nope")))
    assert rc == 2
    assert stubbed["plan_calls"] == []


def test_negative_min_age_is_a_usage_error(tmp_path, stubbed):
    rc = cmd_prune(_make_args(tmp_path, min_age=-1))
    assert rc == 2
    assert stubbed["plan_calls"] == []


def test_json_dry_run_emits_the_plan_on_stdout(tmp_path, stubbed, capsys):
    stubbed["plan"] = ONE_ENV

    rc = cmd_prune(_make_args(tmp_path, dry_run=True, json=True))

    assert rc == 0
    document = json.loads(capsys.readouterr().out)
    assert document["plan"]["envs"][0]["env_name"] == "orb"


def test_json_run_keeps_stdout_clean_even_with_a_confirm(tmp_path, stubbed, monkeypatch, capsys):
    """With --json, stdout is exactly one JSON document: plan, progress, and
    the confirm prompt all ride stderr."""
    stubbed["plan"] = ONE_ENV
    stubbed["report"] = ENV_OK
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda: "y")

    rc = cmd_prune(_make_args(tmp_path, json=True, yes=False))

    assert rc == 0
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document["counts"] == {"ok": 1, "failed": 0, "skipped": 0}
    assert "Proceed with deletion?" in captured.err


def test_prune_overrides_restrictive_umask(tmp_path, stubbed):
    """The manifest and layout marker prune rewrites must stay group-writable,
    whatever the maintainer's personal umask says."""
    old = os.umask(0o077)
    try:
        assert cmd_prune(_make_args(tmp_path, dry_run=True)) == 0
        assert os.umask(0o022) == 0o002
    finally:
        os.umask(old)


def test_planner_errors_are_reported_cleanly(tmp_path, monkeypatch, capsys):
    from rootstock.operations import OperationError

    def exploding_plan(root, **kwargs):
        raise OperationError("Unknown checkpoint id(s): nope")

    monkeypatch.setattr("rootstock.commands.prune.plan_prune", exploding_plan)

    rc = cmd_prune(_make_args(tmp_path, checkpoint=["nope"]))

    assert rc == 1
    assert "Unknown checkpoint id" in capsys.readouterr().err
