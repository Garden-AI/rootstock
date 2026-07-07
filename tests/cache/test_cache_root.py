"""Tests for cache_root separability — install root vs. model-weight cache root."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rootstock.calculator import RootstockCalculator
from rootstock.clusters import (
    CLUSTER_REGISTRY,
    Cluster,
    get_cache_root_for_cluster,
    get_cluster,
    get_root_for_cluster,
)
from rootstock.commands.common import resolve_cache_root
from rootstock.environment import (
    EnvironmentManager,
    get_model_cache_env,
    get_user_cache_dir,
)

# Runtime write-back caches that must NEVER point into the shared roots —
# non-maintainers can only read there.
_PER_USER_VARS = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH",
    "WARP_CACHE_PATH",
    "PYTHONPYCACHEPREFIX",
    "XDG_CONFIG_HOME",
    "MPLCONFIGDIR",
    "CACHED_PATH_CACHE_ROOT",
)

# ---------- get_model_cache_env: pure function ----------------------------


def test_get_model_cache_env_default_unchanged():
    """Regression guard for clusters that don't split (Della, Sophia)."""
    out = get_model_cache_env(Path("/install"))
    assert out["HOME"] == "/install/home"
    assert out["XDG_CACHE_HOME"] == "/install/cache"
    assert out["HF_HOME"] == "/install/cache/huggingface"
    assert out["HF_HUB_CACHE"] == "/install/cache/huggingface/hub"


def test_get_model_cache_env_explicit_cache_root_splits_paths():
    out = get_model_cache_env(Path("/install"), cache_root=Path("/cache"))
    assert out["HOME"] == "/cache/home"
    assert out["XDG_CACHE_HOME"] == "/cache/cache"
    assert out["HF_HOME"] == "/cache/cache/huggingface"
    assert out["HF_HUB_CACHE"] == "/cache/cache/huggingface/hub"


def test_get_model_cache_env_none_cache_root_falls_back_to_root():
    """Passing cache_root=None must be identical to omitting it."""
    a = get_model_cache_env(Path("/install"), cache_root=None)
    b = get_model_cache_env(Path("/install"))
    assert a == b


# ---------- per-user write-back redirection --------------------------------


def test_write_back_caches_never_point_into_shared_roots(monkeypatch, tmp_path: Path):
    for var in _PER_USER_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROOTSTOCK_USER_CACHE_DIR", str(tmp_path / "user-cache"))

    out = get_model_cache_env(Path("/install"), cache_root=Path("/cache"))
    for var in _PER_USER_VARS:
        assert out[var].startswith(str(tmp_path / "user-cache")), f"{var} = {out[var]}"
        assert not out[var].startswith("/install")
        assert not out[var].startswith("/cache")


def test_write_back_caches_respect_preexisting_values(monkeypatch):
    """A user pointing e.g. Triton at node-local SSD in a job script wins."""
    monkeypatch.setenv("TRITON_CACHE_DIR", "/local/scratch/triton")
    out = get_model_cache_env(Path("/install"))
    assert out["TRITON_CACHE_DIR"] == "/local/scratch/triton"


def test_write_back_caches_ignore_empty_preexisting_values(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ROOTSTOCK_USER_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_DIR", "")
    out = get_model_cache_env(Path("/install"))
    assert out["TRITON_CACHE_DIR"] == str(tmp_path / "triton")


def test_get_user_cache_dir_defaults_to_real_home(monkeypatch):
    monkeypatch.delenv("ROOTSTOCK_USER_CACHE_DIR", raising=False)
    assert get_user_cache_dir() == Path.home() / ".cache" / "rootstock"


def test_get_user_cache_dir_env_override(monkeypatch):
    monkeypatch.setenv("ROOTSTOCK_USER_CACHE_DIR", "/pscratch/u/me/rs")
    assert get_user_cache_dir() == Path("/pscratch/u/me/rs")


def test_shared_read_redirects_unaffected_by_user_cache(monkeypatch, tmp_path: Path):
    """Weights are still found in the shared cache — only writes move."""
    monkeypatch.setenv("ROOTSTOCK_USER_CACHE_DIR", str(tmp_path))
    out = get_model_cache_env(Path("/install"), cache_root=Path("/cache"))
    assert out["HOME"] == "/cache/home"
    assert out["HF_HUB_CACHE"] == "/cache/cache/huggingface/hub"


# ---------- Cluster registry ---------------------------------------------


def test_cluster_default_cache_root_is_install_root():
    c = Cluster(root=Path("/x"))
    assert c.resolved_cache_root == Path("/x")


def test_cluster_explicit_cache_root_distinct_from_install_root():
    c = Cluster(root=Path("/install"), cache_root=Path("/cache"))
    assert c.resolved_cache_root == Path("/cache")
    assert c.root != c.cache_root


def test_della_has_no_cache_root_split():
    """Regression guard: Della's behavior is unchanged."""
    della = get_cluster("della")
    assert della.cache_root is None
    assert della.resolved_cache_root == della.root


def test_perlmutter_registered_with_split():
    pm = get_cluster("perlmutter")
    assert pm.cache_root is not None
    assert pm.cache_root != pm.root
    # CFS for code, PSCRATCH for cache.
    assert "cfs" in str(pm.root).lower()
    assert "pscratch" in str(pm.cache_root).lower()


def test_polaris_shares_sophia_eagle_root():
    """Both ALCF machines mount Eagle and share one install."""
    assert get_cluster("polaris").root == get_cluster("sophia").root
    assert get_cluster("polaris").cache_root is None


def test_get_root_for_cluster_returns_install_root():
    assert get_root_for_cluster("perlmutter") == get_cluster("perlmutter").root


def test_get_cache_root_for_cluster_returns_cache_root():
    pm_cache = get_cache_root_for_cluster("perlmutter")
    assert pm_cache == get_cluster("perlmutter").cache_root


def test_get_cache_root_falls_back_to_install_root_for_unsplit_cluster():
    della_cache = get_cache_root_for_cluster("della")
    della_root = get_root_for_cluster("della")
    assert della_cache == della_root


def test_unknown_cluster_raises():
    with pytest.raises(ValueError, match="Unknown cluster"):
        get_cluster("not-a-real-cluster")


# ---------- resolve_cache_root (CLI helper) -------------------------------


def test_resolve_cache_root_for_known_cluster_with_split():
    pm = get_cluster("perlmutter")
    assert resolve_cache_root(pm.root) == pm.cache_root


def test_resolve_cache_root_for_known_cluster_without_split():
    della = get_cluster("della")
    assert resolve_cache_root(della.root) == della.root


def test_resolve_cache_root_for_unknown_root_returns_root():
    """A custom --root path with no registered cluster falls back to itself."""
    custom = Path("/some/random/path")
    assert resolve_cache_root(custom) == custom


# ---------- EnvironmentManager wiring -------------------------------------


def test_environment_manager_passes_cache_root_through(tmp_path: Path):
    (tmp_path / "envs" / "fake_env").mkdir(parents=True)
    cache_dir = tmp_path / "alt_cache"
    mgr = EnvironmentManager(root=tmp_path, cache_root=cache_dir)
    env_vars = mgr.get_environment_variables()
    assert env_vars["XDG_CACHE_HOME"] == str(cache_dir / "cache")
    assert env_vars["HOME"] == str(cache_dir / "home")
    mgr.cleanup()


def test_environment_manager_default_cache_root_uses_install_root(tmp_path: Path):
    (tmp_path / "envs" / "fake_env").mkdir(parents=True)
    mgr = EnvironmentManager(root=tmp_path)
    env_vars = mgr.get_environment_variables()
    assert env_vars["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    mgr.cleanup()


# ---------- RootstockCalculator wiring ------------------------------------


_MACE_ENV_SOURCE = '''\
"""MACE env."""

CHECKPOINTS = {
    "mace-mp-0-medium": "medium",
}


def setup(checkpoint, device="cuda"):
    return None
'''


def _make_mace_env(install: Path) -> None:
    env_dir = install / "envs" / "mace"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    (env_dir / "env_source.py").write_text(_MACE_ENV_SOURCE)


@pytest.fixture
def fake_pm_root(tmp_path: Path, monkeypatch) -> Path:
    """Pretend Perlmutter's CFS root and PSCRATCH cache live under tmp_path,
    so the calculator's existence-check passes without touching real /global/cfs."""
    install = tmp_path / "install"
    cache = tmp_path / "cache"
    install.mkdir()
    _make_mace_env(install)
    cache.mkdir()

    monkeypatch.setitem(
        CLUSTER_REGISTRY,
        "_test_split",
        Cluster(root=install, cache_root=cache),
    )
    return install, cache


def test_calculator_resolves_cache_root_from_cluster(fake_pm_root):
    install, cache = fake_pm_root
    calc = RootstockCalculator(checkpoint="mace-mp-0-medium", cluster="_test_split")
    assert calc.root == install
    assert calc.cache_root == cache


def test_calculator_explicit_cache_root_overrides_cluster_default(fake_pm_root):
    install, cache = fake_pm_root
    override = install.parent / "override_cache"
    override.mkdir()
    calc = RootstockCalculator(
        checkpoint="mace-mp-0-medium",
        cluster="_test_split",
        cache_root=override,
    )
    assert calc.cache_root == override


def test_calculator_with_root_only_defaults_cache_root_to_root(tmp_path: Path):
    _make_mace_env(tmp_path)

    calc = RootstockCalculator(checkpoint="mace-mp-0-medium", root=tmp_path)
    assert calc.cache_root == tmp_path


def test_calculator_with_root_and_explicit_cache_root(tmp_path: Path):
    install = tmp_path / "install"
    install.mkdir()
    _make_mace_env(install)
    cache = tmp_path / "cache"
    cache.mkdir()

    calc = RootstockCalculator(
        checkpoint="mace-mp-0-medium",
        root=install,
        cache_root=cache,
    )
    assert calc.root == install
    assert calc.cache_root == cache


# ---------- End-to-end: cache_root reaches the worker wrapper -------------


def _extract_kwargs_path(wrapper_text: str) -> Path:
    match = re.search(r'open\("([^"]+\.json)"\)', wrapper_text)
    assert match
    return Path(match.group(1))


def test_environment_manager_get_environment_variables_reflects_split(tmp_path: Path):
    """The env vars handed to the worker subprocess use the split cache_root."""
    (tmp_path / "envs" / "fake_env").mkdir(parents=True)
    cache_dir = tmp_path / "alt_cache"
    mgr = EnvironmentManager(root=tmp_path, cache_root=cache_dir)

    env_vars = mgr.get_environment_variables()
    # All four cache-related vars must point under cache_dir, not tmp_path.
    for key in ("HOME", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE"):
        assert env_vars[key].startswith(str(cache_dir)), f"{key} = {env_vars[key]}"
    mgr.cleanup()
