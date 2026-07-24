"""``rootstock usage report`` / ``usage compact`` / ``usage push`` command adapters."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rootstock.commands.usage import cmd_usage_compact, cmd_usage_push, cmd_usage_report
from rootstock.config import UserConfig
from rootstock.manifest import create_manifest, save_manifest
from rootstock.usage import record_session, usage_dir


def _args(tmp_path, **extra):
    base = {"root": str(tmp_path), "cache_root": None, "json": False, "dry_run": False}
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
        client="calculator",
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


# --------------------------------------------------------------------------- #
# push
# --------------------------------------------------------------------------- #

_PUSH_CONFIG = UserConfig(
    api_key="wk-key",
    api_secret="ws-secret",
    api_url="https://garden-ai-dev--rootstock-admin-manifest.modal.run",
)


def _seed_manifest(tmp_path, config=_PUSH_CONFIG, cluster="testcluster"):
    save_manifest(create_manifest(tmp_path, cluster, config), tmp_path)


def _patch_config(monkeypatch, config=_PUSH_CONFIG):
    monkeypatch.setattr("rootstock.commands.usage.load_config", lambda: config)


def test_push_without_credentials_fails(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch, UserConfig())
    assert cmd_usage_push(_args(tmp_path)) == 1
    assert "API key not configured" in capsys.readouterr().err


def test_push_without_derivable_url_fails(tmp_path, monkeypatch, capsys):
    _patch_config(
        monkeypatch,
        UserConfig(api_key="k", api_secret="s", api_url="https://example.com/ingest"),
    )
    assert cmd_usage_push(_args(tmp_path)) == 1
    assert "usage_api_url" in capsys.readouterr().err


def test_push_without_manifest_fails(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch)
    _seed(tmp_path)
    assert cmd_usage_push(_args(tmp_path)) == 1
    assert "manifest init" in capsys.readouterr().err


def test_push_without_spool_fails(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch)
    _seed_manifest(tmp_path)
    assert cmd_usage_push(_args(tmp_path)) == 1
    assert "usage collection is off" in capsys.readouterr().err


def test_push_empty_spool_is_a_noop(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch)
    _seed_manifest(tmp_path)
    usage_dir(tmp_path).mkdir()
    assert cmd_usage_push(_args(tmp_path)) == 0
    assert "nothing to push" in capsys.readouterr().out


def test_push_dry_run_prints_url_and_payload(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch)
    _seed_manifest(tmp_path)
    _seed(tmp_path)
    assert cmd_usage_push(_args(tmp_path, dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "https://garden-ai-dev--rootstock-admin-usage.modal.run" in out
    payload = json.loads(out.split(":\n", 1)[1])
    assert payload["cluster"] == "testcluster"
    assert payload["rows"][0]["checkpoint"] == "mace-mp-0-medium"
    assert "users" not in payload["rows"][0]  # counts only, never hashes


def test_push_posts_rollup_rows(tmp_path, monkeypatch, capsys):
    _patch_config(monkeypatch)
    _seed_manifest(tmp_path)
    _seed(tmp_path)

    posted = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        posted["url"] = request.full_url
        posted["headers"] = request.headers
        posted["body"] = json.loads(request.data)
        return _Response()

    monkeypatch.setattr("rootstock.client.urlopen", fake_urlopen)

    assert cmd_usage_push(_args(tmp_path)) == 0
    assert posted["url"] == "https://garden-ai-dev--rootstock-admin-usage.modal.run"
    assert posted["headers"]["Modal-key"] == "wk-key"
    assert posted["body"]["cluster"] == "testcluster"
    assert posted["body"]["rows"][0]["sessions"] == 1
    assert "users" not in posted["body"]["rows"][0]
    # The success line names the endpoint: pushing to the wrong deployment
    # (dev vs prod) otherwise looks identical to success.
    assert "rootstock-admin-usage" in capsys.readouterr().out
