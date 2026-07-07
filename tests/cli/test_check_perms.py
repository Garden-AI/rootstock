"""Tests for ``rootstock check-perms``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import rootstock.perms as perms
from rootstock.commands.check_perms import cmd_check_perms


def _args(**overrides):
    base = dict(
        root=None,
        cache_root=None,
        cluster=None,
        group=None,
        json=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_clean_root(tmp_path: Path) -> Path:
    root = tmp_path / "rootstock"
    root.mkdir()
    os.chmod(root, 0o2775)
    return root


def _world_traversable_ancestors(monkeypatch):
    # tmp_path lives under per-user 700 directories on most systems; the
    # ancestor walk would flag those and drown out what the test asserts.
    monkeypatch.setattr(perms, "_check_ancestors", lambda path: [])


def test_clean_root_exits_zero(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    _world_traversable_ancestors(monkeypatch)
    root = _make_clean_root(tmp_path)

    rc = cmd_check_perms(_args(root=str(root)))
    assert rc == 0
    assert "OK: no permission issues found." in capsys.readouterr().out


def test_issues_exit_one_and_are_listed(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    _world_traversable_ancestors(monkeypatch)
    root = tmp_path / "rootstock"
    root.mkdir()
    os.chmod(root, 0o700)

    rc = cmd_check_perms(_args(root=str(root)))
    assert rc == 1
    out = capsys.readouterr().out
    assert "not world-readable" in out
    assert "setup-perms" in out
    # No ancestor issues, so no ancestor advice.
    assert "facilities ticket" not in out


def test_restricted_ancestor_is_flagged(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    parent = tmp_path / "project"
    root = parent / "rootstock"
    root.mkdir(parents=True)
    os.chmod(root, 0o2775)
    os.chmod(parent, 0o750)

    rc = cmd_check_perms(_args(root=str(root)))
    assert rc == 1
    out = capsys.readouterr().out
    assert "not world-traversable" in out
    assert "facilities ticket" in out


def test_json_report(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    _world_traversable_ancestors(monkeypatch)
    root = _make_clean_root(tmp_path)

    rc = cmd_check_perms(_args(root=str(root), json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["issues"] == []
    assert payload["install_root"] == str(root)


def test_cluster_resolves_split_roots(monkeypatch):
    checked = {}

    def fake_check(install_root, cache_root=None, **kwargs):
        checked["install"] = install_root
        checked["cache"] = cache_root
        return []

    monkeypatch.setattr("rootstock.commands.check_perms.check_permissions", fake_check)

    rc = cmd_check_perms(_args(cluster="perlmutter"))
    assert rc == 0
    assert checked["install"] == Path("/global/cfs/cdirs/m4845/rootstock")
    assert checked["cache"] == Path("/pscratch/sd/w/wengler/rootstock-cache")


def test_unknown_cluster_errors(capsys):
    rc = cmd_check_perms(_args(cluster="nope"))
    assert rc == 2
    assert "Unknown cluster" in capsys.readouterr().err


def test_missing_root_and_cluster_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        "rootstock.commands.check_perms.load_config",
        lambda: SimpleNamespace(root=None),
    )
    rc = cmd_check_perms(_args())
    assert rc == 2
    assert "install root" in capsys.readouterr().err
