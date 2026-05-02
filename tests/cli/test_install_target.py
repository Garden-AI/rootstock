"""Tests for _rootstock_install_target — picks PyPI vs source path."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from rootstock.commands.install import _rootstock_install_target


def _fake_dist(direct_url: dict | None, version: str = "0.5.0") -> MagicMock:
    dist = MagicMock()
    dist.version = version
    dist.read_text.return_value = json.dumps(direct_url) if direct_url is not None else None
    return dist


def test_returns_pypi_pin_when_no_direct_url():
    """Normal pip install — no direct_url.json, pin to version."""
    with patch(
        "rootstock.commands.install.distribution",
        return_value=_fake_dist(direct_url=None, version="0.5.0"),
    ):
        assert _rootstock_install_target() == "rootstock==0.5.0"


def test_returns_pypi_pin_when_direct_url_not_editable():
    """Installed from a URL/wheel but not editable — still pin to version."""
    with patch(
        "rootstock.commands.install.distribution",
        return_value=_fake_dist(
            direct_url={"url": "file:///some/wheel.whl", "dir_info": {"editable": False}},
            version="0.5.0",
        ),
    ):
        assert _rootstock_install_target() == "rootstock==0.5.0"


def test_returns_source_path_when_editable():
    """Editable install — return source path so workers match driver code."""
    with patch(
        "rootstock.commands.install.distribution",
        return_value=_fake_dist(
            direct_url={
                "url": "file:///pscratch/sd/w/wengler/repos/rootstock",
                "dir_info": {"editable": True},
            },
            version="0.5.0",
        ),
    ):
        assert _rootstock_install_target() == "/pscratch/sd/w/wengler/repos/rootstock"


def test_returns_pypi_pin_when_editable_but_non_file_url():
    """Editable from a vcs URL — fall back to PyPI pin (we can't pip-install vcs path directly here)."""
    with patch(
        "rootstock.commands.install.distribution",
        return_value=_fake_dist(
            direct_url={
                "url": "git+https://github.com/example/rootstock@main",
                "dir_info": {"editable": True},
            },
            version="0.5.0",
        ),
    ):
        assert _rootstock_install_target() == "rootstock==0.5.0"
