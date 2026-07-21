"""How ``rootstock install`` treats the declared model-weight cache root.

Where the cache lives is a deployment-time decision made by ``rootstock
init``; install only backfills the declaration when it is missing. These
tests pin that boundary — a rebuild must never re-point a deployment.
The build itself is stubbed out.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rootstock.commands.install import cmd_install
from rootstock.layout import read_declared_cache_root, write_layout_marker


def _args(root, **overrides):
    base = dict(
        source="mace",
        root=str(root),
        models=None,
        force=False,
        upgrade=False,
        verbose=False,
        no_push=True,
        no_perm_check=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def stub_build(monkeypatch):
    """Neutralize everything except the cache-root bookkeeping."""
    monkeypatch.setattr("rootstock.environment.check_uv_available", lambda: True)
    monkeypatch.setattr(
        "rootstock.commands.install.install_environment",
        lambda *a, **kw: None,
    )


def test_backfills_the_install_root_when_undeclared(tmp_path, stub_build):
    assert cmd_install(_args(tmp_path)) == 0
    assert read_declared_cache_root(tmp_path) == tmp_path


def test_existing_declaration_is_preserved(tmp_path, stub_build):
    write_layout_marker(tmp_path, cache_root="/declared/cache")
    assert cmd_install(_args(tmp_path)) == 0
    assert read_declared_cache_root(tmp_path) == Path("/declared/cache")


def test_repeated_installs_do_not_drift(tmp_path, stub_build):
    write_layout_marker(tmp_path, cache_root="/declared/cache")
    for _ in range(3):
        assert cmd_install(_args(tmp_path)) == 0
    assert read_declared_cache_root(tmp_path) == Path("/declared/cache")


def test_install_rejects_a_cache_root_flag():
    """Regression guard: the cache location is not a per-build override.

    `init` owns it. If this starts passing, a rebuild can silently re-point a
    deployment's weights — see the comment in cmd_install.
    """
    import subprocess

    out = subprocess.run(
        ["uv", "run", "rootstock", "install", "mace", "--cache-root", "/tmp/x"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 2
    assert "unrecognized arguments: --cache-root" in out.stderr


def test_permission_check_uses_the_declared_cache_root(tmp_path, monkeypatch, stub_build):
    """The pre-build warning must stat the split cache root, not the install root."""
    write_layout_marker(tmp_path, cache_root="/declared/cache")
    seen = {}

    def fake_check(install_root, cache_root=None, **kwargs):
        seen["install"] = install_root
        seen["cache"] = cache_root
        return []

    monkeypatch.setattr("rootstock.perms.check_permissions", fake_check)

    cmd_install(_args(tmp_path, no_perm_check=False))
    assert seen["install"] == tmp_path
    assert seen["cache"] == Path("/declared/cache")
