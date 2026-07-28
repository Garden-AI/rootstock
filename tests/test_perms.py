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
        "mkdir -p /install/root/usage",
        "chgrp m4845 /install/root",
        "setfacl -m g:m4845:rwx /install/root",
        "setfacl -dm g:m4845:rwx /install/root",
        "setfacl -dm o::r-X /install/root",
        "chmod 2775 /install/root",
        "chmod 1777 /install/root/usage",
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


def test_render_spool_lives_on_the_cache_half():
    """The spool takes runtime writes from user jobs, so it belongs on the
    cache filesystem — never under an install root like Frontier's /sw."""
    cmds = render_commands("/install/root", cache_root="/cache/root", group="m4845")
    lines = [format_command(c) for c in cmds]
    assert "mkdir -p /cache/root/usage" in lines
    assert "chmod 1777 /cache/root/usage" in lines
    assert not any("/install/root/usage" in line for line in lines)


def test_render_no_usage_spool_omits_it():
    cmds = render_commands("/install/root", group="m4845", usage_spool=False)
    lines = [format_command(c) for c in cmds]
    assert not any("usage" in line for line in lines)


def test_render_spool_chmod_is_last_even_with_retrofit():
    """The retrofit setfacl -R / find pass rewrites modes under the root, so
    the spool's 1777 must be asserted after all of it — same reasoning as the
    setgid-vs-setfacl ordering for the roots themselves."""
    cmds = render_commands("/install/root", group="m4845", retrofit=True)
    lines = [format_command(c) for c in cmds]
    assert lines[-1] == "chmod 1777 /install/root/usage"


def test_render_usage_dir_redirects_the_spool():
    """--usage-dir puts the real 1777 directory somewhere the maintainer
    permanently controls and symlinks {cache_root}/usage to it — for clusters
    like Delta where write access to the install is granted temporarily."""
    cmds = render_commands("/install/root", group="m4845", usage_dir="/home/maint/rs-usage")
    lines = [format_command(c) for c in cmds]
    assert lines[0] == "mkdir -p /home/maint/rs-usage"
    assert lines[1] == "ln -sfn /home/maint/rs-usage /install/root/usage"
    # The mode belongs to the real directory, not the symlink, and still
    # comes last.
    assert lines[-1] == "chmod 1777 /home/maint/rs-usage"
    assert "mkdir -p /install/root/usage" not in lines


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
# usage spool
# --------------------------------------------------------------------------- #


def test_check_missing_spool_is_not_an_issue(tmp_path: Path, monkeypatch):
    """No usage/ dir is how an install opts out of usage collection —
    check-perms must not nag about a deliberate choice."""
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o2775)

    assert check_permissions(root) == []


def test_check_spool_with_correct_mode_is_clean(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o2775)
    spool = root / "usage"
    spool.mkdir()
    os.chmod(spool, 0o1777)

    assert check_permissions(root) == []


def test_check_spool_flags_not_world_writable_and_no_sticky(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o2775)
    spool = root / "usage"
    spool.mkdir()
    os.chmod(spool, 0o755)  # the umask-022 default a bare mkdir would leave

    problems = " ".join(i.problem for i in check_permissions(root))
    assert "not world-writable" in problems
    assert "sticky" in problems


def test_check_spool_looked_up_on_the_cache_half(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    install = tmp_path / "install"
    cache = tmp_path / "cache"
    install.mkdir()
    cache.mkdir()
    os.chmod(install, 0o2775)
    os.chmod(cache, 0o2755)
    spool = cache / "usage"
    spool.mkdir()
    os.chmod(spool, 0o777)  # world-writable but missing the sticky bit

    issues = check_permissions(install, cache)
    assert [i.path for i in issues] == [spool]
    assert "sticky" in issues[0].problem
    # A stray usage/ under the install root is not the spool.
    (install / "usage").mkdir()
    os.chmod(install / "usage", 0o755)
    assert [i.path for i in check_permissions(install, cache)] == [spool]


def test_check_redirected_spool_is_checked_through_the_link(tmp_path: Path, monkeypatch):
    """A --usage-dir redirect is a symlink; the mode rules apply to its
    target, and check-perms follows it there."""
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o2775)
    target = tmp_path / "home-spool"
    target.mkdir()
    os.chmod(target, 0o1777)
    (root / "usage").symlink_to(target)

    assert check_permissions(root) == []

    os.chmod(target, 0o755)  # revoked-access aftermath: target lost its mode
    problems = " ".join(i.problem for i in check_permissions(root))
    assert "not world-writable" in problems


def test_check_dangling_spool_symlink_is_flagged(tmp_path: Path, monkeypatch):
    """A dangling redirect means someone turned collection on and its target
    vanished — unlike a missing spool, that is not a deliberate opt-out."""
    monkeypatch.setattr(perms, "_run_getfacl", lambda path: None)
    root = tmp_path / "root"
    root.mkdir()
    os.chmod(root, 0o2775)
    (root / "usage").symlink_to(tmp_path / "vanished")

    issues = check_permissions(root)
    assert [i.path for i in issues] == [root / "usage"]
    assert "dangling" in issues[0].problem


def test_records_write_through_a_redirected_spool(tmp_path: Path):
    """The Delta scenario end-to-end: {cache_root}/usage is a symlink into a
    directory the maintainer permanently controls, and sessions write
    through it without knowing."""
    from rootstock.usage import record_session

    target = tmp_path / "home-spool"
    target.mkdir()
    (tmp_path / "usage").symlink_to(target)

    path = record_session(
        root=tmp_path,
        cache_root=tmp_path,
        env_name="mace",
        checkpoint="mace-mp-0-medium",
        device="cuda",
        client="calculator",
        started_at="2026-07-23T01:02:03+00:00",
        duration_s=1.0,
        n_calculations=1,
    )

    assert path is not None
    assert (target / path.name).is_file()  # the bytes live in the target


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
