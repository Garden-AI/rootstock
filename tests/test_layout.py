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
    read_declared_cache_root,
    read_layout_version,
    resolve_cache_root,
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
    (tmp_path / "layout.json").write_text(json.dumps({"layout_version": LAYOUT_VERSION + 1}))
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


# --- self-describing cache root ------------------------------------------------


def test_cache_root_declaration_round_trip(tmp_path: Path):
    write_layout_marker(tmp_path, cache_root="/pscratch/whatever/cache")
    assert read_declared_cache_root(tmp_path) == Path("/pscratch/whatever/cache")


def test_no_declaration_reads_as_none(tmp_path: Path):
    write_layout_marker(tmp_path)
    assert read_declared_cache_root(tmp_path) is None


def test_rewrite_without_cache_root_preserves_declaration(tmp_path: Path):
    """A marker rewrite by a command that doesn't know the cache root (or an
    older client) must not erase the install's declaration."""
    write_layout_marker(tmp_path, cache_root="/cache/elsewhere")
    write_layout_marker(tmp_path)
    assert read_declared_cache_root(tmp_path) == Path("/cache/elsewhere")


def test_rewrite_with_same_cache_root_is_a_noop(tmp_path: Path):
    write_layout_marker(tmp_path, cache_root="/cache/elsewhere")
    before = (tmp_path / "layout.json").read_text()

    write_layout_marker(tmp_path, cache_root="/cache/elsewhere")

    assert (tmp_path / "layout.json").read_text() == before


def test_declaration_can_be_updated(tmp_path: Path):
    write_layout_marker(tmp_path, cache_root="/cache/old")
    write_layout_marker(tmp_path, cache_root="/cache/new")
    assert read_declared_cache_root(tmp_path) == Path("/cache/new")


def test_resolve_explicit_override_wins(tmp_path: Path):
    write_layout_marker(tmp_path, cache_root="/cache/declared")
    assert resolve_cache_root(tmp_path, explicit="/cache/explicit") == Path("/cache/explicit")


def test_resolve_prefers_declaration_over_registry(tmp_path: Path, monkeypatch):
    """A pinned client's stale registry must lose to what the install says
    about itself."""
    from rootstock.clusters import CLUSTER_REGISTRY, Cluster

    monkeypatch.setitem(
        CLUSTER_REGISTRY,
        "faketest",
        Cluster(root=tmp_path, cache_root=Path("/registry/stale-cache")),
    )
    write_layout_marker(tmp_path, cache_root="/cache/declared")

    assert resolve_cache_root(tmp_path) == Path("/cache/declared")


def test_resolve_falls_back_to_registry_for_legacy_roots(tmp_path: Path, monkeypatch):
    from rootstock.clusters import CLUSTER_REGISTRY, Cluster

    monkeypatch.setitem(
        CLUSTER_REGISTRY,
        "faketest",
        Cluster(root=tmp_path, cache_root=Path("/registry/cache")),
    )

    assert resolve_cache_root(tmp_path) == Path("/registry/cache")


def test_resolve_defaults_to_root_for_unknown_installs(tmp_path: Path):
    assert resolve_cache_root(tmp_path) == tmp_path


def test_cli_and_calculator_share_one_resolver():
    """The CLI-facing name is the same function object — the two entry
    points cannot diverge again."""
    from rootstock.commands.common import resolve_cache_root as cli_resolver

    assert cli_resolver is resolve_cache_root
