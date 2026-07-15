"""On-disk layout versioning ({root}/layout.json).

The layout is a forever contract: 1.0-era roots will outlive many client
versions. The marker lets a future client refuse a layout it doesn't
understand instead of misreading the tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock.layout import (
    LAYOUT_VERSION,
    ensure_layout_compatible,
    read_layout_version,
    write_layout_marker,
)


def test_write_then_read_round_trip(tmp_path: Path):
    write_layout_marker(tmp_path)
    assert read_layout_version(tmp_path) == LAYOUT_VERSION


def test_marker_records_provenance(tmp_path: Path):
    write_layout_marker(tmp_path)
    data = json.loads((tmp_path / "layout.json").read_text())
    assert data["layout_version"] == LAYOUT_VERSION
    assert data["written_by"].startswith("rootstock ")
    assert data["written_at"]


def test_missing_marker_reads_as_none(tmp_path: Path):
    """Legacy (pre-marker) installs have no layout.json; that's layout 1,
    not an error."""
    assert read_layout_version(tmp_path) is None
    ensure_layout_compatible(tmp_path)  # must not raise


def test_corrupt_marker_reads_as_none(tmp_path: Path):
    (tmp_path / "layout.json").write_text("{not json")
    assert read_layout_version(tmp_path) is None
    ensure_layout_compatible(tmp_path)  # a broken metadata file can't brick the root


def test_current_layout_is_compatible(tmp_path: Path):
    write_layout_marker(tmp_path)
    ensure_layout_compatible(tmp_path)  # must not raise


def test_newer_layout_tells_user_to_upgrade(tmp_path: Path):
    (tmp_path / "layout.json").write_text(
        json.dumps({"layout_version": LAYOUT_VERSION + 1})
    )
    with pytest.raises(RuntimeError, match="upgrade this client"):
        ensure_layout_compatible(tmp_path)


def test_rewrite_is_a_noop_when_current(tmp_path: Path):
    """Repeated installs must not churn the marker (its written_at is the
    first-stamp time, and shared-fs writes aren't free)."""
    write_layout_marker(tmp_path)
    before = (tmp_path / "layout.json").read_text()

    write_layout_marker(tmp_path)

    assert (tmp_path / "layout.json").read_text() == before


def test_stale_marker_is_upgraded(tmp_path: Path):
    (tmp_path / "layout.json").write_text(json.dumps({"layout_version": 0}))
    write_layout_marker(tmp_path)
    assert read_layout_version(tmp_path) == LAYOUT_VERSION
