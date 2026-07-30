"""Shared-install permission recipe: rendering, applying, and verifying.

A rootstock install on an HPC cluster should be **world-readable** — nothing in
it is sensitive (all derived from public PyPI packages and public model
checkpoints), so any cluster user, not just the maintainer's project group,
should be able to use it. Maintainer secrets live in the maintainer's
``~/.config/rootstock/config.toml``, never in the shared root.

This module is the single source of truth for what "correct permissions" means.
``render_commands`` produces the recipe the ``setup-perms`` command renders or
applies; ``check_permissions`` is the bounded, non-recursive verification that
``rootstock install`` runs up front to warn when a root looks misconfigured,
and that ``rootstock check-perms`` exposes as a standalone command (with
ancestor-directory checks added).

The recipe:

* Install root — ``chmod 2775`` (setgid + group-write, world r-x) + ``chgrp
  <group>`` + a named-group ACL (``setfacl -m`` / ``-dm g:<group>:rwx``) so
  co-maintainers in the project group can write and new files inherit that.
* Cache root (only when on a separate filesystem) — ``chmod 2755`` + ``chgrp
  <group>``. The mode bits already give group r-x and world r-x; the setgid bit
  handles group-ownership inheritance. No named-group ACL (maintainer-only-write
  on the cache is the accepted default).
* Usage spool — ``mkdir -p {cache_root}/usage`` + ``chmod 1777`` (sticky,
  /tmp-style): any user's session can drop an anonymous usage record, nobody
  can delete anyone else's, and the dir owner (whoever ran setup-perms) can
  still prune everything when collecting. Existence of the directory is what
  turns usage collection on for an install, so ``--no-usage-spool`` skips it
  and a *missing* spool is never flagged as an issue — only a present one
  with the wrong mode is. ``--usage-dir <path>`` redirects the spool: the
  real 1777 directory is created at ``<path>`` and ``{cache_root}/usage``
  becomes a symlink to it — for clusters where the maintainer's write access
  to the install is temporary (e.g. Delta), so collection keeps working from
  a directory they permanently control (their home, say) after access lapses.
* ``--retrofit`` — recursive ``setfacl -R`` variants so an install that already
  has files in it becomes world-readable, not just future files, plus a
  ``find -type d ... chmod g+s`` so existing subdirectories inherit too.

The ``chmod`` comes *last* in the rendered order: setting an ACL rewrites a
path's mode bits, which can clear setgid, so the mode is asserted after all the
``setfacl`` work rather than before it.
"""

from __future__ import annotations

import shlex
import stat as stat_module
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .usage import usage_dir

# Mode bits for each root. setgid (2) + rwx owner + rwx/r-x group + r-x other.
INSTALL_ROOT_MODE = "2775"
CACHE_ROOT_MODE = "2755"
# The usage spool: sticky + world-writable, like /tmp. Its location comes
# from usage.py's usage_dir(), so provisioning and the writer can't disagree.
USAGE_SPOOL_MODE = "1777"


# --------------------------------------------------------------------------- #
# Rendering / applying the recipe
# --------------------------------------------------------------------------- #


def render_commands(
    install_root: Path | str,
    cache_root: Path | str | None = None,
    *,
    group: str,
    retrofit: bool = False,
    usage_spool: bool = True,
    usage_dir: Path | str | None = None,
) -> list[list[str]]:
    """Render the permission recipe as a list of argv lists.

    The cache-root commands are emitted only when ``cache_root`` is given and
    differs from ``install_root`` (a single-filesystem cluster needs nothing
    extra for the cache). ``retrofit`` appends the recursive variants.
    ``usage_spool`` provisions the world-writable usage-record spool (its
    existence opts the install into usage collection). ``usage_dir``
    redirects it: the real directory is created (and moded) there, and
    ``{cache_root}/usage`` becomes a symlink to it — for clusters where the
    maintainer's write access to the install is temporary (e.g. Delta), so
    the spool lives somewhere they keep control of, like their home.
    """
    install_root = Path(install_root)
    cmds: list[list[str]] = [
        ["chgrp", group, str(install_root)],
        ["setfacl", "-m", f"g:{group}:rwx", str(install_root)],
        ["setfacl", "-dm", f"g:{group}:rwx", str(install_root)],
        # State the default ``other`` entry rather than letting setfacl derive
        # it from the mode: the chmod runs last (see below), so at this point
        # the mode may still be the restrictive one we're fixing, and a derived
        # ``other::---`` default would leave new files unreadable to everyone
        # outside the group.
        ["setfacl", "-dm", "o::r-X", str(install_root)],
    ]

    cache_path = Path(cache_root) if cache_root is not None else install_root
    separate_cache = cache_path != install_root
    # The spool lives on the cache half of the install — that's the
    # runtime-writable side (on Frontier the install root is under /sw, which
    # must not take writes from user jobs).
    spool = usage_spool_dir(install_root, cache_root)
    # With a redirect, the path every writer and reader uses is unchanged —
    # {cache_root}/usage — it just resolves through a symlink to a directory
    # the maintainer keeps control of. -sfn so re-running setup-perms is
    # idempotent (replaces a stale link; fails loudly if a real directory is
    # already in the way rather than silently nesting).
    spool_target = Path(usage_dir) if usage_dir is not None else spool
    if usage_spool:
        cmds.insert(0, ["mkdir", "-p", str(spool_target)])
        if spool_target != spool:
            cmds.insert(1, ["ln", "-sfn", str(spool_target), str(spool)])

    if separate_cache:
        cmds += [
            ["chgrp", group, str(cache_path)],
            # Same reasoning, and it's what makes newly downloaded weights
            # world-readable regardless of the maintainer's umask. No
            # named-group entry — maintainer-only-write on the cache is the
            # accepted default.
            ["setfacl", "-dm", "o::r-X", str(cache_path)],
        ]

    if retrofit:
        # Existing files: make the named-group ACL and world r-x apply to what's
        # already there, plus default ACLs so the tree stays consistent.
        # Capital ``X`` grants execute only on directories (and already-executable
        # files), so recursing doesn't mark every .py/.so/weight file executable.
        cmds += [
            ["setfacl", "-R", "-m", f"g:{group}:rwX", str(install_root)],
            ["setfacl", "-R", "-dm", f"g:{group}:rwX", str(install_root)],
            ["setfacl", "-R", "-m", "o::r-X", str(install_root)],
            ["setfacl", "-R", "-dm", "o::r-X", str(install_root)],
        ]
        if separate_cache:
            cmds += [
                ["setfacl", "-R", "-m", "o::r-X", str(cache_path)],
                ["setfacl", "-R", "-dm", "o::r-X", str(cache_path)],
            ]
        # setgid has to hold on every *existing* directory too, not just the
        # root: a subdirectory without it hands new files the creator's primary
        # group. ``-exec ... +`` batches, so this is one chmod per few thousand
        # dirs rather than one per dir.
        cmds += [_setgid_dirs(install_root)]
        if separate_cache:
            cmds += [_setgid_dirs(cache_path)]

    # Mode bits go *last*, deliberately. Setting an ACL rewrites the file mode
    # (the ACL's owner/mask/other entries are the mode bits), and on some
    # filesystems that drops the setgid bit — observed on NERSC CFS in 2026-07,
    # where a chmod-first recipe left the root at 0775 and check-perms flagged
    # it immediately after a successful setup-perms --apply. Re-asserting the
    # mode after every setfacl is cheap and makes the recipe order-independent.
    cmds += [["chmod", INSTALL_ROOT_MODE, str(install_root)]]
    if separate_cache:
        cmds += [["chmod", CACHE_ROOT_MODE, str(cache_path)]]
    if usage_spool:
        # After the retrofit setfacl/find pass, which would otherwise rewrite
        # the spool's mode along with everything else under the root. Moded
        # on the target: with a redirect the mode belongs to the real
        # directory, not the symlink.
        cmds += [["chmod", USAGE_SPOOL_MODE, str(spool_target)]]

    return cmds


def usage_spool_dir(install_root: Path | str, cache_root: Path | str | None = None) -> Path:
    """Where an install's usage spool lives: under the cache root when the
    install has a separate one, else under the install root itself.

    Delegates to usage.py's ``usage_dir`` — this wrapper only supplies the
    perms-side convention that a missing cache_root means the install root
    doubles as the cache half."""
    return usage_dir(cache_root if cache_root is not None else install_root)


def _setgid_dirs(root: Path) -> list[str]:
    """``find``-based recursive setgid for directories under ``root``."""
    return ["find", str(root), "-type", "d", "-exec", "chmod", "g+s", "{}", "+"]


def format_command(argv: list[str]) -> str:
    """Render an argv list as a copy-pasteable, shell-quoted line."""
    return " ".join(shlex.quote(arg) for arg in argv)


# --------------------------------------------------------------------------- #
# Verifying existing permissions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PermIssue:
    """A single thing that looks wrong about a root's permissions.

    ``ancestor`` distinguishes issues on directories *above* a root (fixable
    only by whoever owns them) from issues on the root itself (fixable with
    ``rootstock setup-perms``), so callers can give the right advice.
    """

    path: Path
    problem: str
    ancestor: bool = False


def _world_readable_traversable(mode: int) -> bool:
    """True if ``other`` has both read and execute (traverse) bits."""
    return (mode & 0o5) == 0o5


def _check_ancestors(path: Path) -> list[PermIssue]:
    """Flag ancestor directories that block world traversal to ``path``.

    A perfectly world-readable install is unreachable if any directory above
    it lacks ``o+x`` (e.g., a project parent directory that was created
    group-only). One stat per path component — cheap on HPC filesystems.
    """
    issues: list[PermIssue] = []
    for parent in reversed(path.resolve().parents):
        if parent == Path("/"):
            continue
        try:
            mode = parent.stat().st_mode
        except FileNotFoundError:
            break  # the missing-root issue is reported by _check_root
        except PermissionError:
            issues.append(
                PermIssue(
                    parent,
                    "cannot stat ancestor (permission denied for you, too)",
                    ancestor=True,
                )
            )
            break
        if not (mode & 0o1):
            issues.append(
                PermIssue(
                    parent,
                    "ancestor not world-traversable (other lacks x); "
                    "users outside the project group cannot reach the install",
                    ancestor=True,
                )
            )
    return issues


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


def _check_usage_spool(install_root: Path, cache_root: Path | None) -> list[PermIssue]:
    """Flag a usage spool that exists but can't do its job.

    A *missing* spool is not an issue — its absence is how an install opts
    out of usage collection. A present one, though, must be world-writable
    (or users' sessions silently fail to record) and sticky (or any user can
    delete everyone else's records). A redirected spool (a symlink from
    setup-perms --usage-dir) is checked through the link — same rules apply
    to the target — except a *dangling* link, which is flagged: someone set
    up collection and its target has since vanished.
    """
    spool = usage_spool_dir(install_root, cache_root)
    try:
        mode = spool.stat().st_mode
    except FileNotFoundError:
        if spool.is_symlink():
            return [
                PermIssue(
                    spool,
                    "usage spool is a dangling symlink (redirect target is "
                    "missing); users' sessions can't record usage",
                )
            ]
        return []
    except PermissionError:
        return []

    issues: list[PermIssue] = []
    if not stat_module.S_ISDIR(mode):
        return [PermIssue(spool, "usage spool is not a directory")]
    if (mode & 0o007) != 0o007:
        issues.append(
            PermIssue(
                spool,
                f"usage spool not world-writable (mode should be {USAGE_SPOOL_MODE}); "
                "users' sessions can't record usage",
            )
        )
    if not (mode & stat_module.S_ISVTX):
        issues.append(
            PermIssue(
                spool,
                f"usage spool lacks the sticky bit (mode should be {USAGE_SPOOL_MODE}); "
                "any user can delete other users' records",
            )
        )
    return issues


def check_permissions(
    install_root: Path | str,
    cache_root: Path | str | None = None,
    *,
    group: str | None = None,
    include_ancestors: bool = False,
) -> list[PermIssue]:
    """Best-effort, bounded permission check for an install (and cache) root.

    Touches only the root directories themselves — never recurses downward —
    so it is cheap even on slow HPC filesystems. Returns a (possibly empty)
    list of issues; callers treat these as warnings, not errors.

    ``include_ancestors`` additionally checks that every directory above each
    root grants world traverse (``o+x``). Off by default: dev installs under a
    ``700`` home directory are legitimate, so only the explicit pre-launch
    check (``rootstock check-perms``) opts in.
    """
    install_root = Path(install_root)
    issues: list[PermIssue] = []
    if include_ancestors:
        issues += _check_ancestors(install_root)
    issues += _check_root(install_root, kind="install", group=group, require_group_acl=True)

    if cache_root is not None and Path(cache_root) != install_root:
        cache_root = Path(cache_root)
        if include_ancestors:
            seen = {issue.path for issue in issues}
            issues += [i for i in _check_ancestors(cache_root) if i.path not in seen]
        issues += _check_root(cache_root, kind="cache", group=group, require_group_acl=False)

    issues += _check_usage_spool(install_root, Path(cache_root) if cache_root else None)

    return issues
