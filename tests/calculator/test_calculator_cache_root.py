"""RootstockCalculator resolves cache_root from the install itself.

Historically the two entry points disagreed: CLI `--root` reverse-looked-up
the cluster registry (so Perlmutter got its PSCRATCH cache) while
``RootstockCalculator(root=...)`` silently defaulted cache_root to the
install root. Both now resolve through ``rootstock.layout.resolve_cache_root``:
explicit override > the install's own declaration in layout.json > registry
fallback for legacy roots > the root itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.calculator import RootstockCalculator
from rootstock.clusters import CLUSTER_REGISTRY, Cluster
from rootstock.layout import write_layout_marker

_ENV_SOURCE = """\
CHECKPOINTS = {
    "mace-mp-0-medium": "medium",
}


def setup(checkpoint, device="cuda"):
    return None
"""


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    env_dir = tmp_path / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_ENV_SOURCE)
    return tmp_path


def _calc(**kwargs) -> RootstockCalculator:
    return RootstockCalculator(checkpoint="mace-mp-0-medium", device="cpu", **kwargs)


def test_root_entry_point_honors_install_declaration(fake_root: Path):
    write_layout_marker(fake_root, cache_root="/declared/cache")

    calc = _calc(root=fake_root)

    assert calc.cache_root == Path("/declared/cache")


def test_explicit_cache_root_overrides_declaration(fake_root: Path):
    write_layout_marker(fake_root, cache_root="/declared/cache")

    calc = _calc(root=fake_root, cache_root="/explicit/cache")

    assert calc.cache_root == Path("/explicit/cache")


def test_unknown_root_without_declaration_defaults_to_root(fake_root: Path):
    calc = _calc(root=fake_root)

    assert calc.cache_root == fake_root


def test_cluster_entry_point_honors_install_declaration(fake_root: Path, monkeypatch):
    """cluster= is only a name->path bootstrap; the install's declaration
    beats the registry's (possibly stale) cache_root."""
    monkeypatch.setitem(
        CLUSTER_REGISTRY,
        "faketest",
        Cluster(root=fake_root, cache_root=Path("/registry/stale-cache")),
    )
    write_layout_marker(fake_root, cache_root="/declared/cache")

    calc = _calc(cluster="faketest")

    assert calc.root == fake_root
    assert calc.cache_root == Path("/declared/cache")


def test_cluster_entry_point_falls_back_to_registry_for_legacy_roots(fake_root: Path, monkeypatch):
    monkeypatch.setitem(
        CLUSTER_REGISTRY,
        "faketest",
        Cluster(root=fake_root, cache_root=Path("/registry/cache")),
    )

    calc = _calc(cluster="faketest")

    assert calc.cache_root == Path("/registry/cache")


def test_root_entry_point_matches_cli_for_registered_legacy_roots(fake_root: Path, monkeypatch):
    """The historical divergence: root= must reverse-look-up the registry
    exactly like CLI --root does, not silently default to the install root."""
    monkeypatch.setitem(
        CLUSTER_REGISTRY,
        "faketest",
        Cluster(root=fake_root, cache_root=Path("/registry/cache")),
    )

    calc = _calc(root=fake_root)

    assert calc.cache_root == Path("/registry/cache")
