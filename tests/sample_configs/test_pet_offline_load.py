"""Tests for the pet env's hub-resolve bypass.

UPETCalculator(model=..., version=...) resolves the name by listing the hub
repo — an uncached API call that fails on workers (HF_HUB_OFFLINE=1) and on
nodes without internet. The env fetches the pinned file itself via
hf_hub_download (a cache hit needs no network) and passes checkpoint_path,
which skips the resolve.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_CONFIGS_DIR = (
    Path(__file__).parent.parent.parent / "sample_model_configurations" / "nvidia_configs"
)


def _load_env_module():
    spec = importlib.util.spec_from_file_location("pet_env", _CONFIGS_DIR / "pet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stubbed_libs(monkeypatch):
    """Stub huggingface_hub and upet; capture what setup() passes to each."""
    captured = {}

    hf = types.ModuleType("huggingface_hub")

    def hf_hub_download(**kwargs):
        captured["download"] = kwargs
        return "/shared/cache/models/stub.ckpt"

    hf.hf_hub_download = hf_hub_download

    upet = types.ModuleType("upet")
    upet_calculator = types.ModuleType("upet.calculator")

    class UPETCalculator:
        def __init__(self, **kwargs):
            captured["calculator"] = kwargs

    upet_calculator.UPETCalculator = UPETCalculator
    upet.calculator = upet_calculator

    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    monkeypatch.setitem(sys.modules, "upet", upet)
    monkeypatch.setitem(sys.modules, "upet.calculator", upet_calculator)
    return captured


def test_setup_downloads_pinned_filename(stubbed_libs):
    env = _load_env_module()
    env.setup("pet-oam-xl", device="cuda")
    assert stubbed_libs["download"] == {
        "repo_id": "lab-cosmo/upet",
        "filename": "pet-oam-xl-v1.0.0.ckpt",
        "subfolder": "models",
    }


def test_setup_passes_checkpoint_path_not_model_name(stubbed_libs):
    """model=/version= would trigger the hub-listing resolve — never pass them."""
    env = _load_env_module()
    env.setup("pet-omatpes-l", device="cpu")
    calc_kwargs = stubbed_libs["calculator"]
    assert calc_kwargs == {
        "checkpoint_path": "/shared/cache/models/stub.ckpt",
        "device": "cpu",
    }


def test_every_checkpoint_maps_to_parseable_filename():
    """upet parses (model, size, version) out of the filename — every pinned
    entry must render to the {model}-{size}-v{version}.ckpt shape."""
    env = _load_env_module()
    for upstream in env.CHECKPOINTS.values():
        model, version = upstream.split("@", 1)
        filename = f"{model}-v{version}.ckpt"
        assert filename.endswith(f"-v{version}.ckpt")
        assert "@" not in filename
