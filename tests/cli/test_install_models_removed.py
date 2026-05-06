"""``rootstock install --models`` is removed; verify it errors loudly."""

from __future__ import annotations

import subprocess
import sys


def test_install_models_emits_migration_error():
    """The CLI should reject --models and point at 'rootstock add'."""
    result = subprocess.run(
        [sys.executable, "-m", "rootstock.cli", "install", "/tmp/whatever.py", "--models", "small"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "rootstock add" in result.stderr
