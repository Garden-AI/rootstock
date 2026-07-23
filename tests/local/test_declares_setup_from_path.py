"""Tests for environment.declares_setup_from_path."""

from __future__ import annotations

from rootstock.environment import declares_setup_from_path

_WITH_HOOK = """\
CHECKPOINTS = {"a-1": "a"}


def setup(checkpoint, device="cuda"):
    return None


def setup_from_path(path, device="cuda", **kwargs):
    return None
"""

_WITHOUT_HOOK = """\
CHECKPOINTS = {"a-1": "a"}


def setup(checkpoint, device="cuda"):
    return None
"""

_NESTED_ONLY = """\
CHECKPOINTS = {"a-1": "a"}


def setup(checkpoint, device="cuda"):
    def setup_from_path(path):
        return None

    return None
"""


def test_detects_module_level_hook(tmp_path):
    src = tmp_path / "env_source.py"
    src.write_text(_WITH_HOOK)
    assert declares_setup_from_path(src) is True


def test_absent_hook(tmp_path):
    src = tmp_path / "env_source.py"
    src.write_text(_WITHOUT_HOOK)
    assert declares_setup_from_path(src) is False


def test_nested_definition_does_not_count(tmp_path):
    src = tmp_path / "env_source.py"
    src.write_text(_NESTED_ONLY)
    assert declares_setup_from_path(src) is False
