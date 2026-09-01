from __future__ import annotations

from pathlib import Path

import pytest
from stagelib import build_install_root


@pytest.fixture
def install_root(tmp_path: Path) -> Path:
    return build_install_root(tmp_path)
