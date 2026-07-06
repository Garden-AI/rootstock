"""Tests for the orb env files' cached_path bypass (#67).

At serve time the worker must write nothing under the shared root. orb-models
routes its default weights URL through `cached_path`, which write-locks its
cache dir even on warm hits — so the orb env files pre-fetch the checkpoint
into the shared model cache at `rootstock add` time and hand the loader a
local `weights_path`, which cached_path returns without locking.

The env files' module level is stdlib-only, so they import without
torch/orb-models installed.
"""

from __future__ import annotations

import importlib.util
import io
import urllib.request
from pathlib import Path

import pytest

_CONFIGS_DIR = (
    Path(__file__).parent.parent.parent / "sample_model_configurations" / "nvidia_configs"
)


def _load_env_module(name: str):
    spec = importlib.util.spec_from_file_location(name, _CONFIGS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["orb", "orb_v3"])
def env_module(request):
    return _load_env_module(request.param)


# ---------- _default_weights_url -------------------------------------------


def test_default_weights_url_reads_signature_default(env_module):
    def load_fn(
        weights_path: str = "https://example.com/orb-v2-20241011.ckpt",
        device=None,
    ):
        pass

    assert env_module._default_weights_url(load_fn) == "https://example.com/orb-v2-20241011.ckpt"


def test_default_weights_url_rejects_non_url_default(env_module):
    def load_fn(weights_path: str = "/some/local/default.ckpt", device=None):
        pass

    with pytest.raises(RuntimeError, match="no URL default"):
        env_module._default_weights_url(load_fn)


def test_default_weights_url_fails_loudly_without_weights_path_param(env_module):
    """If a future orb-models drops the kwarg, populate/serve must not silently
    fall back to the locking cached_path route."""

    def load_fn(device=None):
        pass

    with pytest.raises(KeyError):
        env_module._default_weights_url(load_fn)


# ---------- _local_weights_path ---------------------------------------------


def test_local_weights_path_lands_in_shared_cache(env_module, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/shared/cache")
    out = env_module._local_weights_path("https://example.com/models/orb-v2-20241011.ckpt")
    assert out == Path("/shared/cache/orb/orb-v2-20241011.ckpt")


def test_local_weights_path_falls_back_to_home_cache(env_module, monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    out = env_module._local_weights_path("https://example.com/orb-v2.ckpt")
    assert out == Path.home() / ".cache" / "orb" / "orb-v2.ckpt"


def test_local_weights_path_ignores_empty_xdg_cache_home(env_module, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    out = env_module._local_weights_path("https://example.com/orb-v2.ckpt")
    assert out == Path.home() / ".cache" / "orb" / "orb-v2.ckpt"


# ---------- _fetch ----------------------------------------------------------


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_fetch_downloads_atomically(env_module, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url: _FakeResponse(b"checkpoint-bytes")
    )
    dest = tmp_path / "orb" / "orb-v2.ckpt"
    env_module._fetch("https://example.com/orb-v2.ckpt", dest)

    assert dest.read_bytes() == b"checkpoint-bytes"
    # No temp litter next to the final file.
    assert list(dest.parent.iterdir()) == [dest]


def test_fetch_cleans_up_partial_download_on_failure(
    env_module, monkeypatch, tmp_path: Path
):
    def _boom(url):
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    dest = tmp_path / "orb" / "orb-v2.ckpt"
    with pytest.raises(OSError, match="network down"):
        env_module._fetch("https://example.com/orb-v2.ckpt", dest)

    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []


# ---------- serve-time invariant: setup() end-to-end ------------------------

_FAKE_URL = "https://example.com/models/fake-orb.ckpt"


def _install_fake_orb_models(monkeypatch, env_name: str, calls: dict):
    """Stub torch + orb_models in sys.modules so setup() runs without them."""
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.device = lambda spec: f"device({spec})"

    def load_fn(
        weights_path: str = _FAKE_URL,
        device=None,
        precision: str = "float32-high",
    ):
        calls["weights_path"] = weights_path
        if env_name == "orb_v3":
            return "orbff", "adapter"
        return "orbff"

    pretrained = types.ModuleType("orb_models.forcefield.pretrained")
    # One loader per canonical id, orb-models style (orb_v2, orb_v3_..., etc.).
    for fn_name in ("orb_v2", "orb_v3_conservative_inf_omat"):
        setattr(pretrained, fn_name, load_fn)

    class ORBCalculator:
        def __init__(self, orbff, atoms_adapter=None, device=None):
            calls["calculator"] = (orbff, atoms_adapter, device)

    calculator = types.ModuleType("orb_models.forcefield.calculator")
    calculator.ORBCalculator = ORBCalculator
    inference_calculator = types.ModuleType("orb_models.forcefield.inference.calculator")
    inference_calculator.ORBCalculator = ORBCalculator

    forcefield = types.ModuleType("orb_models.forcefield")
    forcefield.pretrained = pretrained
    inference = types.ModuleType("orb_models.forcefield.inference")
    orb_models = types.ModuleType("orb_models")

    for name, module in {
        "torch": torch,
        "orb_models": orb_models,
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.pretrained": pretrained,
        "orb_models.forcefield.calculator": calculator,
        "orb_models.forcefield.inference": inference,
        "orb_models.forcefield.inference.calculator": inference_calculator,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


_FIRST_CHECKPOINT = {"orb": "orb-v2", "orb_v3": "orb-v3-conservative-inf-omat"}


@pytest.fixture(params=["orb", "orb_v3"])
def wired_env(request, monkeypatch):
    calls: dict = {}
    _install_fake_orb_models(monkeypatch, request.param, calls)
    return _load_env_module(request.param), _FIRST_CHECKPOINT[request.param], calls


def test_setup_serves_warm_cache_without_network_or_shared_writes(
    wired_env, monkeypatch, tmp_path: Path
):
    """The #67 invariant: a warm serve hands the loader a local weights file
    and never opens a URL — so cached_path takes no lock under the shared root."""
    module, checkpoint, calls = wired_env
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    weights = module._local_weights_path(_FAKE_URL)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"already here")

    def _no_network(url):
        raise AssertionError(f"urlopen({url!r}) called on a warm cache")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)

    module.setup(checkpoint, device="cpu")
    assert calls["weights_path"] == str(weights)


def test_setup_fetches_weights_on_cold_cache(wired_env, monkeypatch, tmp_path: Path):
    """Populate path (`rootstock add`): cold cache downloads into the shared
    model cache, then loads from the local file."""
    module, checkpoint, calls = wired_env
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url: _FakeResponse(b"downloaded")
    )

    module.setup(checkpoint, device="cpu")

    weights = module._local_weights_path(_FAKE_URL)
    assert weights.read_bytes() == b"downloaded"
    assert calls["weights_path"] == str(weights)
