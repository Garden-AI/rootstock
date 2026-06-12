"""Unit tests for the permission recipe rendering and verification."""

from __future__ import annotations

import os
from pathlib import Path

import rootstock.perms as perms
from rootstock.perms import (
    _parse_getfacl,
    check_permissions,
    format_command,
    render_commands,
)

# --------------------------------------------------------------------------- #
# render_commands
# --------------------------------------------------------------------------- #


def test_render_single_filesystem():
    cmds = render_commands("/install/root", group="m4845")
    lines = [format_command(c) for c in cmds]
    assert lines == [
        "chmod 2775 /install/root",
        "chgrp m4845 /install/root",
        "setfacl -m g:m4845:rwx /install/root",
        "setfacl -dm g:m4845:rwx /install/root",
    ]


def test_render_split_filesystem():
    cmds = render_commands("/install/root", cache_root="/cache/root", group="m4845")
    lines = [format_command(c) for c in cmds]
    # Install-root commands plus the two cache-root mode/group commands.
    assert "chmod 2755 /cache/root" in lines
    assert "chgrp m4845 /cache/root" in lines
    # No named-group ACL on the cache root.
    assert not any("setfacl" in line and "/cache/root" in line for line in lines)


def test_render_cache_root_same_as_install_emits_nothing_extra():
    cmds = render_commands("/install/root", cache_root="/install/root", group="m4845")
    lines = [format_command(c) for c in cmds]
    assert not any("/cache" in line for line in lines)
    assert len(cmds) == 4


def test_render_retrofit_adds_recursive_variants():
    cmds = render_commands("/install/root", cache_root="/cache/root", group="m4845", retrofit=True)
    lines = [format_command(c) for c in cmds]
    # Capital X so recursing doesn't mark every file executable.
    assert "setfacl -R -m g:m4845:rwX /install/root" in lines
    assert "setfacl -R -dm g:m4845:rwX /install/root" in lines
    assert "setfacl -R -m o::r-X /install/root" in lines
    assert "setfacl -R -dm o::r-X /install/root" in lines
    # World-readable retrofit also applies to the separate cache root.
    assert "setfacl -R -m o::r-X /cache/root" in lines
    assert "setfacl -R -dm o::r-X /cache/root" in lines
    # ...but no recursive named-group ACL on the cache root.
    assert "setfacl -R -m g:m4845:rwX /cache/root" not in lines


# --------------------------------------------------------------------------- #
# _parse_getfacl
# --------------------------------------------------------------------------- #


GETFACL_SAMPLE = """\
user::rwx
group::r-x
group:m4845:rwx
mask::rwx
other::r-x
default:user::rwx
default:group::r-x
default:group:m4845:rwx
default:mask::rwx
default:other::r-x
"""

GETFACL_CLAMPED = """\
user::rwx
group::r-x
group:m4845:rwx\t\t\t#effective:r--
mask::r--
other::r-x
"""


def test_parse_getfacl_splits_access_and_default():
    access, default = _parse_getfacl(GETFACL_SAMPLE)
    assert access[("group", "m4845")] == ("rwx", None)
    assert access[("other", "")] == ("r-x", None)
    assert default[("group", "m4845")] == ("rwx", None)
    assert default[("other", "")] == ("r-x", None)


def test_parse_getfacl_captures_effective_clamp():
    access, default = _parse_getfacl(GETFACL_CLAMPED)
    perms_str, effective = access[("group", "m4845")]
    assert perms_str == "rwx"
    assert effective == "r--"
    assert default == {}


# --------------------------------------------------------------------------- #
# check_permissions (stat-based; ACL tooling stubbed off)
# --------------------------------------------------------------------------- #


def test_check_flags_missing_setgid_and_world_read(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o700)  # no world r-x, no setgid

    issues = check_permissions(root)
    problems = " ".join(i.problem for i in issues)
    assert "world-readable" in problems
    assert "setgid" in problems


def test_check_clean_when_mode_bits_correct(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o2775)  # setgid + world r-x

    assert check_permissions(root) == []


def test_check_missing_root(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    issues = check_permissions(tmp_path / "does-not-exist")
    assert any("does not exist" in i.problem for i in issues)


def test_check_acl_flags_missing_default_and_mask_clamp(tmp_path: Path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o2775)
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: GETFACL_CLAMPED)

    issues = check_permissions(root, group="m4845")
    problems = " ".join(i.problem for i in issues)
    assert "no default ACL" in problems
    assert "mask clamps" in problems
