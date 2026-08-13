"""Tests for git+ env-source specs (rootstock.gitsource).

Parsing is pure; fetching is exercised against a throwaway local repository
over the file:// transport (which, unlike a plain path, supports the same
shallow-fetch path used against real remotes).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rootstock.commands.common import resolve_source_arg
from rootstock.gitsource import (
    GitSource,
    fetch_git_source,
    is_git_source,
    parse_git_source,
)
from rootstock.operations import OperationError

# ---------------------------------------------------------------------------
# parsing


def test_is_git_source_only_matches_the_prefix():
    assert is_git_source("git+https://github.com/org/repo.git")
    assert not is_git_source("./staging/delta-environments")
    assert not is_git_source("/abs/path")


def test_parse_url_only():
    parsed = parse_git_source("git+https://github.com/org/repo.git")
    assert parsed == GitSource("https://github.com/org/repo.git", None, None)


def test_parse_ref_and_subdirectory():
    parsed = parse_git_source(
        "git+https://github.com/org/repo.git@v1.3.0#subdirectory=configs/delta"
    )
    assert parsed == GitSource("https://github.com/org/repo.git", "v1.3.0", "configs/delta")


def test_parse_branch_ref_with_slash():
    parsed = parse_git_source("git+https://github.com/org/repo.git@feature/new-envs")
    assert parsed.ref == "feature/new-envs"
    assert parsed.url == "https://github.com/org/repo.git"


def test_parse_ssh_userinfo_is_not_a_ref_separator():
    parsed = parse_git_source("git+ssh://git@github.com/org/repo.git")
    assert parsed == GitSource("ssh://git@github.com/org/repo.git", None, None)

    parsed = parse_git_source("git+ssh://git@github.com/org/repo.git@main")
    assert parsed == GitSource("ssh://git@github.com/org/repo.git", "main", None)


@pytest.mark.parametrize(
    "spec",
    [
        "git+https://github.com/org/repo.git@",  # empty ref
        "git+github.com/org/repo",  # no scheme
        "git+https://github.com/org/repo.git#egg=whatever",  # unknown fragment
    ],
)
def test_parse_rejects_malformed_specs(spec):
    with pytest.raises(OperationError):
        parse_git_source(spec)


def test_parse_rejects_non_git_specs():
    with pytest.raises(OperationError, match="git\\+URL"):
        parse_git_source("./local/dir")


# ---------------------------------------------------------------------------
# fetching (local file:// repository)

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _run(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> dict:
    """A local repo with an environments/ subdir, two commits, and a tag."""
    repo = tmp_path / "upstream"
    (repo / "environments").mkdir(parents=True)
    _run(["init", "--quiet", "-b", "main"], cwd=repo)
    # Serve arbitrary SHAs over the file:// transport, like GitHub does.
    _run(["config", "uploadpack.allowAnySHA1InWant", "true"], cwd=repo)

    (repo / "environments" / "mace.py").write_text("CHECKPOINTS = {}\n")
    (repo / "README.md").write_text("not an env source\n")
    _run(["add", "-A"], cwd=repo)
    _run(["commit", "--quiet", "-m", "first"], cwd=repo)
    _run(["tag", "v1"], cwd=repo)
    first_sha = _run(["rev-parse", "HEAD"], cwd=repo)

    (repo / "environments" / "uma.py").write_text("CHECKPOINTS = {}\n")
    _run(["add", "-A"], cwd=repo)
    _run(["commit", "--quiet", "-m", "second"], cwd=repo)

    return {"url": f"file://{repo}", "first_sha": first_sha}


@requires_git
def test_fetch_default_branch_repo_root(upstream, tmp_path):
    dest = tmp_path / "checkout"
    result = fetch_git_source(GitSource(upstream["url"], None, None), dest)

    assert result == dest
    assert (result / "environments" / "uma.py").exists()


@requires_git
def test_fetch_subdirectory(upstream, tmp_path):
    source = GitSource(upstream["url"], None, "environments")
    result = fetch_git_source(source, tmp_path / "checkout")

    assert result.name == "environments"
    assert sorted(p.name for p in result.glob("*.py")) == ["mace.py", "uma.py"]


@requires_git
def test_fetch_tag_ref(upstream, tmp_path):
    source = GitSource(upstream["url"], "v1", "environments")
    result = fetch_git_source(source, tmp_path / "checkout")

    assert (result / "mace.py").exists()
    assert not (result / "uma.py").exists(), "tag predates the second commit"


@requires_git
def test_fetch_sha_ref(upstream, tmp_path):
    source = GitSource(upstream["url"], upstream["first_sha"], None)
    result = fetch_git_source(source, tmp_path / "checkout")

    assert not (result / "environments" / "uma.py").exists()


@requires_git
def test_fetch_missing_subdirectory_errors(upstream, tmp_path):
    source = GitSource(upstream["url"], None, "no-such-dir")
    with pytest.raises(OperationError, match="no-such-dir"):
        fetch_git_source(source, tmp_path / "checkout")


@requires_git
def test_fetch_escaping_subdirectory_errors(upstream, tmp_path):
    source = GitSource(upstream["url"], None, "../outside")
    with pytest.raises(OperationError, match="escapes"):
        fetch_git_source(source, tmp_path / "checkout")


@requires_git
def test_fetch_bad_ref_surfaces_git_stderr(upstream, tmp_path):
    source = GitSource(upstream["url"], "no-such-ref", None)
    with pytest.raises(OperationError, match="git fetch failed"):
        fetch_git_source(source, tmp_path / "checkout")


# ---------------------------------------------------------------------------
# the shared CLI resolver


def test_resolve_source_arg_local_dir(tmp_path):
    assert resolve_source_arg(str(tmp_path)) == tmp_path


def test_resolve_source_arg_missing_dir_raises(tmp_path):
    with pytest.raises(OperationError, match="not a directory"):
        resolve_source_arg(str(tmp_path / "nope"))


@requires_git
def test_resolve_source_arg_git_spec(upstream):
    result = resolve_source_arg(f"git+{upstream['url']}#subdirectory=environments")
    assert (result / "mace.py").exists()
