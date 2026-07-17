"""The public API surface is a deliberate, pinned decision.

Once 1.0 tags, every top-level export is implicitly semver-protected — so
the blessed list is asserted exactly, and anything that grows it must be a
conscious change to this test.
"""

from __future__ import annotations

import pytest

import rootstock

BLESSED = {
    "RootstockCalculator",
    "RootstockServer",
    "list_declared_checkpoints",
    "CheckpointNotFoundError",
    "CLUSTER_REGISTRY",
    "Cluster",
    "get_cluster",
    "get_root_for_cluster",
}


def test_all_is_exactly_the_blessed_surface():
    assert set(rootstock.__all__) == BLESSED


def test_every_blessed_name_resolves():
    for name in BLESSED:
        assert getattr(rootstock, name) is not None


def test_version_is_available():
    assert isinstance(rootstock.__version__, str)


@pytest.mark.parametrize(
    "name",
    ["run_worker", "save_config", "parse_pep723_metadata", "Manifest", "EnvironmentManager"],
)
def test_pruned_internals_raise_with_guidance(name: str):
    with pytest.raises(AttributeError, match="no longer part of the public rootstock API"):
        getattr(rootstock, name)


def test_unknown_attribute_raises_plain_attributeerror():
    with pytest.raises(AttributeError, match="has no attribute"):
        rootstock.definitely_not_a_name
