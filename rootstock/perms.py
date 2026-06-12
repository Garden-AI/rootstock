"""Shared-install permission recipe: rendering, applying, and verifying.

A rootstock install on an HPC cluster should be **world-readable** — nothing in
it is sensitive (all derived from public PyPI packages and public model
checkpoints), so any cluster user, not just the maintainer's project group,
should be able to use it. Maintainer secrets live in the maintainer's
``~/.config/rootstock/config.toml``, never in the shared root.

This module is the single source of truth for what "correct permissions" means.
``render_commands`` produces the recipe the ``setup-perms`` command renders or
applies; ``check_permissions`` is the bounded, non-recursive verification that
``rootstock install`` runs up front to warn when a root looks misconfigured.

The recipe:

* Install root — ``chmod 2775`` (setgid + group-write, world r-x) + ``chgrp
  <group>`` + a named-group ACL (``setfacl -m`` / ``-dm g:<group>:rwx``) so
  co-maintainers in the project group can write and new files inherit that.
* Cache root (only when on a separate filesystem) — ``chmod 2755`` + ``chgrp
  <group>``. The mode bits already give group r-x and world r-x; the setgid bit
  handles group-ownership inheritance. No named-group ACL (maintainer-only-write
  on the cache is the accepted default).
* ``--retrofit`` — recursive ``setfacl -R`` variants so an install that already
  has files in it becomes world-readable, not just future files.
"""

from __future__ import annotations

import shlex
import stat as stat_module
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Mode bits for each root. setgid (2) + rwx owner + rwx/r-x group + r-x other.
INSTALL_ROOT_MODE = "2775"
CACHE_ROOT_MODE = "2755"


# --------------------------------------------------------------------------- #
# Rendering / applying the recipe
# --------------------------------------------------------------------------- #


def render_commands(
    install_root: Path | str,
    cache_root: Path | str | None = None,
    *,
    group: str,
    retrofit: bool = False,
) -> list[list[str]]:
    """Render the permission recipe as a list of argv lists.

    The cache-root commands are emitted only when ``cache_root`` is given and
    differs from ``install_root`` (a single-filesystem cluster needs nothing
    extra for the cache). ``retrofit`` appends the recursive variants.
    """
    install_root = Path(install_root)
    cmds: list[list[str]] = [
        ["chmod", INSTALL_ROOT_MODE, str(install_root)],
        ["chgrp", group, str(install_root)],
        ["setfacl", "-m", f"g:{group}:rwx", str(install_root)],
        ["setfacl", "-dm", f"g:{group}:rwx", str(install_root)],
    ]

    separate_cache = cache_root is not None and Path(cache_root) != install_root
    if separate_cache:
        cache_root = Path(cache_root)
        cmds += [
            ["chmod", CACHE_ROOT_MODE, str(cache_root)],
            ["chgrp", group, str(cache_root)],
        ]

    if retrofit:
        # Existing files: make the named-group ACL and world r-x apply to what's
        # already there, plus default ACLs so the tree stays consistent.
        cmds += [
            ["setfacl", "-R", "-m", f"g:{group}:rwx", str(install_root)],
            ["setfacl", "-R", "-dm", f"g:{group}:rwx", str(install_root)],
            ["setfacl", "-R", "-m", "o::r-x", str(install_root)],
            ["setfacl", "-R", "-dm", "o::r-x", str(install_root)],
        ]
        if separate_cache:
            cmds += [
                ["setfacl", "-R", "-m", "o::r-x", str(cache_root)],
                ["setfacl", "-R", "-dm", "o::r-x", str(cache_root)],
            ]

    return cmds


def format_command(argv: list[str]) -> str:
    """Render an argv list as a copy-pasteable, shell-quoted line."""
    return " ".join(shlex.quote(arg) for arg in argv)


# --------------------------------------------------------------------------- #
# Verifying existing permissions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PermIssue:
    """A single thing that looks wrong about a root's permissions."""

    path: Path
    problem: str


def _world_readable_traversable(mode: int) -> bool:
    """True if ``other`` has both read and execute (traverse) bits."""
    return (mode & 0o5) == 0o5


def _has_setgid(mode: int) -> bool:
    return bool(mode & stat_module.S_ISGID)


def _gid_name(gid: int) -> str | None:
    try:
        import grp

        return grp.getgrgid(gid).gr_name
    except (KeyError, ImportError):
        return None


def _run_getfacl(path: Path) -> str | None:
    """Return ``getfacl -c`` output for ``path``, or None if unavailable.

    Best-effort: a missing ``getfacl`` (e.g. macOS, which uses a different ACL
    model) or a non-zero exit means we simply skip ACL-level checks.
    """
    try:
        result = subprocess.run(
            ["getfacl", "-c", str(path)],
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_getfacl(output: str) -> tuple[dict, dict]:
    """Parse ``getfacl -c`` output into ``(access, default)`` entry maps.

    Each map is keyed by ``(type, qualifier)`` (e.g. ``("group", "m4845")`` or
    ``("other", "")``) with value ``(perms, effective)`` where ``perms`` is the
    raw ``rwx`` string and ``effective`` is the ``#effective:`` annotation when
    the mask clamps the entry, else ``None``.
    """
    access: dict = {}
    default: dict = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        effective = None
        if "#effective:" in line:
            main, _, eff = line.partition("#effective:")
            line = main.strip()
            effective = eff.strip()

        parts = line.split(":")
        is_default = parts and parts[0] == "default"
        if is_default:
            parts = parts[1:]

        if len(parts) == 3:
            etype, qual, perms = parts
        elif len(parts) == 2:
            etype, perms = parts
            qual = ""
        else:
            continue

        (default if is_default else access)[(etype, qual)] = (perms, effective)

    return access, default


def _perms_have(perms: str, *flags: str) -> bool:
    return all(flag in perms for flag in flags)


def _check_root(
    path: Path,
    *,
    kind: str,
    group: str | None,
    require_group_acl: bool,
) -> list[PermIssue]:
    issues: list[PermIssue] = []

    try:
        st = path.stat()
    except FileNotFoundError:
        return [PermIssue(path, f"{kind} root does not exist")]

    mode = st.st_mode
    if not _world_readable_traversable(mode):
        issues.append(PermIssue(path, "not world-readable/traversable (other lacks r-x)"))
    if not _has_setgid(mode):
        issues.append(
            PermIssue(path, "setgid bit not set; new files won't inherit the project group")
        )

    output = _run_getfacl(path)
    if output is None:
        return issues  # ACL tooling unavailable — stop at the stat-based checks.

    access, default = _parse_getfacl(output)

    if not default:
        issues.append(
            PermIssue(path, "no default ACL; new files won't inherit group-write / world-read")
        )
    else:
        d_other = default.get(("other", ""))
        if d_other is not None and not _perms_have(d_other[0], "r", "x"):
            issues.append(
                PermIssue(
                    path,
                    "default ACL doesn't grant other r-x; new files won't be world-readable",
                )
            )

    # Mask clamp: a group entry whose effective perms were reduced below its
    # granted perms is the classic "maintainer's umask was too restrictive" bug.
    for (etype, qual), (perms, effective) in access.items():
        if etype == "group" and effective is not None and effective != perms:
            label = f"group:{qual}" if qual else "group"
            issues.append(
                PermIssue(
                    path,
                    f"mask clamps {label} from {perms} to {effective} (umask too restrictive?)",
                )
            )
            break

    if require_group_acl:
        expected = group or _gid_name(st.st_gid)
        if expected and ("group", expected) not in access:
            issues.append(
                PermIssue(path, f"no named-group ACL for '{expected}'; co-maintainers can't write")
            )

    return issues


def check_permissions(
    install_root: Path | str,
    cache_root: Path | str | None = None,
    *,
    group: str | None = None,
) -> list[PermIssue]:
    """Best-effort, bounded permission check for an install (and cache) root.

    Touches only the root directories themselves — never recurses — so it is
    cheap even on slow HPC filesystems. Returns a (possibly empty) list of
    issues; callers treat these as warnings, not errors.
    """
    install_root = Path(install_root)
    issues = _check_root(install_root, kind="install", group=group, require_group_acl=True)

    if cache_root is not None and Path(cache_root) != install_root:
        issues += _check_root(Path(cache_root), kind="cache", group=group, require_group_acl=False)

    return issues
