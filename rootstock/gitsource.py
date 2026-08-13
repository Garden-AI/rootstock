"""Fetch env-source directories from git repositories.

``sync`` and ``prune`` take an optional positional source of env definitions.
Besides a local staging directory, that positional accepts a pip-style git
spec::

    git+URL[@REF][#subdirectory=DIR]

    git+https://github.com/Garden-AI/rootstock.git#subdirectory=sample_model_configurations/nvidia_configs
    git+https://github.com/Garden-AI/rootstock.git@v1.3.0#subdirectory=environments
    git+ssh://git@github.com/Garden-AI/rootstock.git@main

The spec is shallow-fetched (one commit, no history) into a temp directory
that lives until process exit — long enough for the planner to hash the
staged files and the executor to register/build from them, after which the
install root holds its own copies and the checkout is disposable.

REF may be a branch, tag, or full commit SHA (whatever the server allows
``fetch`` by name — GitHub allows all three). Omitted REF means the remote's
default branch. Omitted subdirectory means the repository root.
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .operations import OperationError

GIT_SPEC_PREFIX = "git+"


def is_git_source(spec: str) -> bool:
    """Whether a source positional names a git spec rather than a local dir."""
    return spec.startswith(GIT_SPEC_PREFIX)


@dataclass(frozen=True)
class GitSource:
    url: str  # clone URL, scheme intact, ref/fragment stripped
    ref: str | None  # branch, tag, or full SHA; None = remote default branch
    subdirectory: str | None  # repo-relative dir holding the *.py sources


def parse_git_source(spec: str) -> GitSource:
    """Parse ``git+URL[@REF][#subdirectory=DIR]`` into its parts.

    The ``@REF`` separator is an ``@`` in the URL *path* (so the user info in
    ``ssh://git@github.com/...`` never trips it); a ref containing ``/``
    (feature branches) parses fine since we split on the last ``@``.
    """
    if not is_git_source(spec):
        raise OperationError(f"Not a git source spec (expected git+URL): {spec}")

    parts = urlsplit(spec[len(GIT_SPEC_PREFIX) :])
    if not parts.scheme or not (parts.netloc or parts.path):
        raise OperationError(
            f"Malformed git source {spec!r}: expected git+URL[@REF][#subdirectory=DIR] "
            "(scp-style addresses need the ssh:// form, e.g. git+ssh://git@host/org/repo.git)"
        )

    path, ref = parts.path, None
    if "@" in path:
        path, _, ref = path.rpartition("@")
        if not ref:
            raise OperationError(f"Malformed git source {spec!r}: empty ref after '@'")

    subdirectory = None
    if parts.fragment:
        fragment = parse_qs(parts.fragment)
        unknown = sorted(set(fragment) - {"subdirectory"})
        if unknown:
            raise OperationError(
                f"Unknown fragment option(s) in git source: {', '.join(unknown)} "
                "(only 'subdirectory' is supported)"
            )
        subdirectory = fragment.get("subdirectory", [None])[0]

    url = parts._replace(path=path, fragment="").geturl()
    return GitSource(url=url, ref=ref, subdirectory=subdirectory)


def _git(args: list[str], cwd: Path) -> None:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "(no output)"
        raise OperationError(f"git {args[0]} failed: {detail}")


def fetch_git_source(source: GitSource, dest: Path) -> Path:
    """Shallow-fetch ``source`` into ``dest``; return the env-source dir.

    One code path covers branch, tag, and SHA refs: init + ``fetch --depth 1
    origin <ref>`` + detached checkout of FETCH_HEAD (a plain shallow clone
    can't target a SHA).
    """
    if shutil.which("git") is None:
        raise OperationError("git not found in PATH (required for git+ sources)")

    dest.mkdir(parents=True, exist_ok=True)
    _git(["init", "--quiet", "."], cwd=dest)
    _git(["remote", "add", "origin", source.url], cwd=dest)
    _git(["fetch", "--quiet", "--depth", "1", "origin", source.ref or "HEAD"], cwd=dest)
    _git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=dest)

    if source.subdirectory is None:
        return dest

    subdir = (dest / source.subdirectory).resolve()
    if dest.resolve() not in subdir.parents and subdir != dest.resolve():
        raise OperationError(f"subdirectory escapes the repository: {source.subdirectory}")
    if not subdir.is_dir():
        raise OperationError(
            f"subdirectory {source.subdirectory!r} not found in {source.url}"
            f"{f' @ {source.ref}' if source.ref else ''}"
        )
    return subdir


def materialize_git_source(spec: str) -> Path:
    """Resolve a ``git+`` spec to a local directory of env sources.

    The checkout lands in a temp directory removed at process exit — callers
    hold the returned path for at most the life of one CLI command, and
    everything durable is copied into the install root during the run.
    """
    source = parse_git_source(spec)
    checkout = Path(tempfile.mkdtemp(prefix="rootstock-git-src-"))
    atexit.register(shutil.rmtree, checkout, ignore_errors=True)
    return fetch_git_source(source, checkout)
