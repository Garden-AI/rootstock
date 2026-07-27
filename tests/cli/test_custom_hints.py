"""':custom' entries are the discoverability surface: listings show them
verbatim, plus one pattern line — and only when an installed env actually
declares an entry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rootstock.benchmark import list_available
from rootstock.commands.status import cmd_list

_CUSTOM_ENV_SOURCE = """\
CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
    "uma:custom": None,
}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""

# Declares the hook but no ':custom' entry (e.g. a built env predating the
# entries): user weights are unavailable there, so nothing may be advertised.
_HOOK_ONLY_ENV_SOURCE = """\
CHECKPOINTS = {"orb-v2": "orb-v2"}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""


def _make_root(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "root"
    env_dir = root / "envs" / "env1"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(source)
    return root


@pytest.mark.parametrize(
    ("source", "hinted"),
    [(_CUSTOM_ENV_SOURCE, True), (_HOOK_ONLY_ENV_SOURCE, False)],
    ids=["custom-entry-env", "hook-only-env"],
)
def test_cmd_list_hint_follows_entry_availability(tmp_path, capsys, source, hinted):
    root = _make_root(tmp_path, source)
    rc = cmd_list(SimpleNamespace(root=str(root)))
    assert rc == 0
    assert ("uma:custom" in capsys.readouterr().out) is hinted


@pytest.mark.parametrize(
    ("source", "hinted"),
    [(_CUSTOM_ENV_SOURCE, True), (_HOOK_ONLY_ENV_SOURCE, False)],
    ids=["custom-entry-env", "hook-only-env"],
)
def test_benchmark_list_shows_custom_entries(tmp_path, capsys, source, hinted):
    root = _make_root(tmp_path, source)
    rc = list_available(root)
    assert rc == 0
    out = capsys.readouterr().out
    assert ("uma:custom" in out) is hinted
    assert ("--weights" in out) is hinted
