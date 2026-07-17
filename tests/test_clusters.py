"""Cluster registry: a name -> path bootstrap, with an honest reverse lookup."""

from __future__ import annotations

from rootstock.clusters import CLUSTER_REGISTRY, get_cluster_for_root


def test_reverse_lookup_unique_root():
    assert get_cluster_for_root(CLUSTER_REGISTRY["della"].root) == "della"


def test_reverse_lookup_unknown_root_is_none():
    assert get_cluster_for_root("/nonexistent/rootstock") is None


def test_reverse_lookup_shared_root_is_ambiguous():
    """sophia and polaris share one Eagle install; picking one by registry
    order would be a guess, so the lookup must decline to answer."""
    shared = CLUSTER_REGISTRY["sophia"].root
    assert str(CLUSTER_REGISTRY["polaris"].root) == str(shared)  # test premise
    assert get_cluster_for_root(shared) is None
