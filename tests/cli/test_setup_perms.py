"""Tests for ``rootstock setup-perms``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rootstock.commands.setup_perms import cmd_setup_perms
from rootstock.perms import PermIssue


def _args(**overrides):
    base = dict(
        root=None,
        root_flag=None,
        cache_root=None,
        cluster=None,
        group="m4845",
        apply=False,
        retrofit=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_dry_run_single_filesystem(capsys):
    rc = cmd_setup_perms(_args(root="/install/root"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "chmod 2775 /install/root" in out
    assert "chgrp m4845 /install/root" in out
    assert "setfacl -m g:m4845:rwx /install/root" in out
    assert "setfacl -dm g:m4845:rwx /install/root" in out
    # No filesystem was touched and no cache-root commands.
    assert "/cache" not in out


def test_dry_run_split_filesystem(capsys):
    rc = cmd_setup_perms(_args(root="/install/root", cache_root="/cache/root"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "chmod 2755 /cache/root" in out
    assert "chgrp m4845 /cache/root" in out


def test_cluster_resolves_split_roots(capsys):
    rc = cmd_setup_perms(_args(cluster="perlmutter"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "chmod 2775 /global/cfs/cdirs/m5268/rootstock" in out
    assert "chmod 2755 /pscratch/sd/o/oprice/rootstock-cache" in out


def test_cluster_single_root_no_cache_commands(capsys):
    rc = cmd_setup_perms(_args(cluster="della"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "chmod 2775 /scratch/gpfs/ROSENGROUP/common/rootstock" in out
    assert "chmod 2755" not in out


def test_unknown_cluster_errors(capsys):
    rc = cmd_setup_perms(_args(cluster="nope"))
    assert rc == 1
    assert "Unknown cluster" in capsys.readouterr().err


def test_missing_root_and_cluster_errors(capsys):
    rc = cmd_setup_perms(_args())
    assert rc == 1
    assert "install root" in capsys.readouterr().err


def test_retrofit_adds_recursive(capsys):
    rc = cmd_setup_perms(_args(root="/install/root", retrofit=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "setfacl -R -m g:m4845:rwX /install/root" in out
    assert "setfacl -R -dm o::r-X /install/root" in out


def test_apply_runs_commands_after_confirmation(monkeypatch, capsys):
    calls = []

    def fake_run(argv, capture_output=False, text=False):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("rootstock.commands.setup_perms.subprocess.run", fake_run)
    monkeypatch.setattr("rootstock.commands.setup_perms.check_permissions", lambda *a, **k: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = cmd_setup_perms(_args(root="/install/root", apply=True))
    assert rc == 0
    # The mode bits are asserted last, after the ACL work that can clear setgid.
    assert calls[-1] == ["chmod", "2775", "/install/root"]
    assert "Permissions applied." in capsys.readouterr().out


def test_apply_reports_issues_that_survive(monkeypatch, capsys):
    """Every command exiting 0 doesn't mean the roots ended up correct."""
    monkeypatch.setattr(
        "rootstock.commands.setup_perms.subprocess.run",
        lambda argv, capture_output=False, text=False: SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(
        "rootstock.commands.setup_perms.check_permissions",
        lambda *a, **k: [PermIssue(Path("/install/root"), "setgid bit not set")],
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = cmd_setup_perms(_args(root="/install/root", apply=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "still look wrong" in err
    assert "setgid bit not set" in err


def test_apply_accepts_root_flag(monkeypatch, capsys):
    """--root is accepted alongside the positional form."""
    monkeypatch.setattr(
        "rootstock.commands.setup_perms.subprocess.run",
        lambda argv, capture_output=False, text=False: SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr("rootstock.commands.setup_perms.check_permissions", lambda *a, **k: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = cmd_setup_perms(_args(root=None, root_flag="/install/root", apply=True))
    assert rc == 0
    assert "/install/root" in capsys.readouterr().out


def test_apply_aborts_without_confirmation(monkeypatch, capsys):
    def fake_run(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("subprocess.run should not be called when aborting")

    monkeypatch.setattr("rootstock.commands.setup_perms.subprocess.run", fake_run)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    rc = cmd_setup_perms(_args(root="/install/root", apply=True))
    assert rc == 1
    assert "Aborted." in capsys.readouterr().out


def test_apply_bails_on_first_failure(monkeypatch, capsys):
    calls = []

    def fake_run(argv, capture_output=False, text=False):
        calls.append(argv)
        # Fail on the first command (chgrp).
        if argv[0] == "chgrp":
            return SimpleNamespace(returncode=1, stderr="chgrp: invalid group")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("rootstock.commands.setup_perms.subprocess.run", fake_run)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    rc = cmd_setup_perms(_args(root="/install/root", apply=True))
    assert rc == 1
    err = capsys.readouterr().err
    assert "command failed: chgrp m4845 /install/root" in err
    # Stopped at the failure — no setfacl was attempted.
    assert not any(c[0] == "setfacl" for c in calls)
