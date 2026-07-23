"""``rootstock usage report`` / ``usage compact`` command adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rootstock.commands.usage import cmd_usage_compact, cmd_usage_report
from rootstock.usage import record_session, usage_dir


def _args(tmp_path, **extra):
    base = {"root": str(tmp_path), "cache_root": None, "json": False}
    base.update(extra)
    return SimpleNamespace(**base)


def _seed(tmp_path):
    usage_dir(tmp_path).mkdir()
    record_session(
        root=tmp_path,
        cache_root=tmp_path,
        env_name="mace",
        checkpoint="mace-mp-0-medium",
        is_local=False,
        device="cuda",
        started_at="2026-07-23T01:02:03+00:00",
        duration_s=60.0,
        n_calculations=42,
    )


def test_report_without_spool_explains_and_fails(tmp_path, capsys):
    assert cmd_usage_report(_args(tmp_path)) == 1
    err = capsys.readouterr().err
    assert "usage collection is off" in err
    assert "setup-perms" in err


def test_report_prints_aggregated_table(tmp_path, capsys):
    _seed(tmp_path)
    assert cmd_usage_report(_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "mace-mp-0-medium" in out
    assert "2026-07" in out
    assert "42" in out


def test_report_json(tmp_path, capsys):
    _seed(tmp_path)
    assert cmd_usage_report(_args(tmp_path, json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["rows"][0]["n_calculations"] == 42
    assert data["rows"][0]["unique_users"] == 1
    assert data["unique_users"] == 1
    assert data["skipped"] == 0
    assert "users" not in data["rows"][0]  # counts only, never hashes


def test_compact_then_report(tmp_path, capsys):
    _seed(tmp_path)
    assert cmd_usage_compact(_args(tmp_path)) == 0
    assert "Compacted 1 record(s)" in capsys.readouterr().out

    assert cmd_usage_report(_args(tmp_path, json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["rows"][0]["sessions"] == 1
