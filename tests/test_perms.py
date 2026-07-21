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
    # chmod goes last: setting an ACL rewrites the mode and can drop setgid.
    assert lines == [
        "chgrp m4845 /install/root",
        "setfacl -m g:m4845:rwx /install/root",
        "setfacl -dm g:m4845:rwx /install/root",
        "setfacl -dm o::r-X /install/root",
        "chmod 2775 /install/root",
    ]


def test_render_split_filesystem():
    cmds = render_commands("/install/root", cache_root="/cache/root", group="m4845")
    lines = [format_command(c) for c in cmds]
    # Mode, group, and a default ACL so new weights are born world-readable.
    assert "chmod 2755 /cache/root" in lines
    assert "chgrp m4845 /cache/root" in lines
    assert "setfacl -dm o::r-X /cache/root" in lines
    # ...but no named-group ACL: maintainer-only-write on the cache is the
    # accepted default.
    assert not any("g:m4845" in line and "/cache/root" in line for line in lines)


def test_render_cache_root_same_as_install_emits_nothing_extra():
    cmds = render_commands("/install/root", cache_root="/install/root", group="m4845")
    lines = [format_command(c) for c in cmds]
    assert not any("/cache" in line for line in lines)
    assert lines == [format_command(c) for c in render_commands("/install/root", group="m4845")]


def test_render_chmod_follows_every_setfacl():
    """Every path's chmod must come after the last setfacl touching that path.

    A setfacl can rewrite the mode bits (and drop setgid on some filesystems),
    so a chmod-first recipe leaves the root without setgid — the NERSC CFS bug.
    """
    for retrofit in (False, True):
        cmds = render_commands(
            "/install/root", cache_root="/cache/root", group="m4845", retrofit=retrofit
        )
        for root in ("/install/root", "/cache/root"):
            touching = [i for i, c in enumerate(cmds) if root in c]
            chmods = [i for i in touching if cmds[i][0] == "chmod"]
            setfacls = [i for i in touching if cmds[i][0] == "setfacl"]
            assert chmods, f"no chmod for {root} (retrofit={retrofit})"
            assert all(i < min(chmods) for i in setfacls), (
                f"setfacl runs after chmod for {root} (retrofit={retrofit})"
            )


def test_render_covers_what_the_checker_demands_of_every_root():
    """Both roots get a default ACL granting other r-x.

    ``_check_root`` reports "no default ACL" / "default ACL doesn't grant other
    r-x" for the cache root as well as the install root, so a recipe that skips
    the cache leaves setup-perms --apply reporting issues it never tried to fix.
    """
    cmds = render_commands("/install/root", cache_root="/cache/root", group="m4845")
    for root in ("/install/root", "/cache/root"):
        defaults = [c for c in cmds if c[0] == "setfacl" and "-dm" in c and root in c]
        assert any("o::r-X" in c for c in defaults), f"no default other ACL for {root}"


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


def test_render_retrofit_sets_setgid_on_existing_dirs():
    cmds = render_commands("/install/root", cache_root="/cache/root", group="m4845", retrofit=True)
    lines = [format_command(c) for c in cmds]
    assert "find /install/root -type d -exec chmod g+s '{}' +" in lines
    assert "find /cache/root -type d -exec chmod g+s '{}' +" in lines
    # Without --retrofit only the root itself is touched.
    plain = [format_command(c) for c in render_commands("/install/root", group="m4845")]
    assert not any(line.startswith("find ") for line in plain)


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


# --------------------------------------------------------------------------- #
# ancestor traversal
# --------------------------------------------------------------------------- #


def test_ancestor_lacking_world_x_flagged(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    parent = tmp_path / "project"
    root = parent / "rootstock"
    root.mkdir(parents=True)
    os.chmod(root, 0o2775)
    os.chmod(parent, 0o750)  # the ALCF failure mode: project dir blocks outsiders

    issues = check_permissions(root, include_ancestors=True)
    assert any(i.path == parent.resolve() and "not world-traversable" in i.problem for i in issues)


def test_ancestors_not_checked_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    parent = tmp_path / "project"
    root = parent / "rootstock"
    root.mkdir(parents=True)
    os.chmod(root, 0o2775)
    os.chmod(parent, 0o750)

    assert check_permissions(root) == []


def test_shared_ancestors_reported_once_for_split_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    parent = tmp_path / "project"
    install = parent / "rootstock"
    cache = parent / "rootstock-cache"
    install.mkdir(parents=True)
    cache.mkdir()
    os.chmod(install, 0o2775)
    os.chmod(cache, 0o2755)
    os.chmod(parent, 0o750)

    issues = check_permissions(install, cache, include_ancestors=True)
    flagged = [i for i in issues if i.path == parent.resolve()]
    assert len(flagged) == 1
