"""Tests for the sample configs' ``setup_from_path`` hooks (local checkpoints).

The envs shipped for both vendors (present in nvidia_configs *and*
amd_configs) are the canonical set for `rootstock add-local`: both copies
must declare the hook, with the contract signature
``setup_from_path(path, device="cuda", **extras-with-defaults)``, and the
signatures must not drift between vendor copies.

The env files' module level is stdlib-only, so they import without torch or
the model packages installed; the orb functional tests stub those modules.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from rootstock.environment import declares_setup_from_path

_SAMPLES = Path(__file__).parent.parent.parent / "sample_model_configurations"
_NVIDIA = _SAMPLES / "nvidia_configs"
_AMD = _SAMPLES / "amd_configs"

_DUAL_VENDOR_ENVS = sorted(p.stem for p in _AMD.glob("*.py") if (_NVIDIA / p.name).exists())


def _hook_args(config: Path) -> ast.arguments | None:
    tree = ast.parse(config.read_text(), filename=str(config))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "setup_from_path":
            return node.args
    return None


# ---------- parity + signature contract --------------------------------------


def test_dual_vendor_set_is_nonempty():
    """Guard against the glob silently matching nothing after a reorg."""
    assert _DUAL_VENDOR_ENVS


@pytest.mark.parametrize("vendor_dir", [_NVIDIA, _AMD], ids=["nvidia", "amd"])
@pytest.mark.parametrize("env", _DUAL_VENDOR_ENVS)
def test_dual_vendor_envs_declare_hook(env, vendor_dir):
    assert declares_setup_from_path(vendor_dir / f"{env}.py"), (
        f"{vendor_dir.name}/{env}.py must declare setup_from_path — every "
        f"dual-vendor env supports local checkpoints"
    )


@pytest.mark.parametrize("vendor_dir", [_NVIDIA, _AMD], ids=["nvidia", "amd"])
@pytest.mark.parametrize("env", _DUAL_VENDOR_ENVS)
def test_hook_signature_contract(env, vendor_dir):
    """First param `path`, then `device` with a default, extras all defaulted
    — so `add-local` without --kwarg works for every env."""
    args = _hook_args(vendor_dir / f"{env}.py")
    names = [a.arg for a in args.args]
    assert names[:2] == ["path", "device"]
    # Everything after `path` has a default.
    assert len(args.defaults) == len(args.args) - 1


@pytest.mark.parametrize("env", _DUAL_VENDOR_ENVS)
def test_hook_signature_matches_across_vendors(env):
    nvidia = _hook_args(_NVIDIA / f"{env}.py")
    amd = _hook_args(_AMD / f"{env}.py")
    assert ast.dump(nvidia) == ast.dump(amd), (
        f"setup_from_path signature drifted between nvidia_configs/{env}.py "
        f"and amd_configs/{env}.py"
    )


# ---------- orb: the one hook with logic (arch routing) -----------------------


def _load_env_module(configs_dir: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        f"{configs_dir.name}_{name}", configs_dir / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_orb_models(monkeypatch, calls: dict):
    """Stub torch + orb_models in sys.modules so the hook runs without them."""
    torch = types.ModuleType("torch")
    torch.device = lambda spec: f"device({spec})"

    pretrained = types.ModuleType("orb_models.forcefield.pretrained")

    def orb_v2(weights_path=None, device=None):
        calls["weights_path"] = weights_path
        calls["load_device"] = device
        return "orbff"

    pretrained.orb_v2 = orb_v2

    class ORBCalculator:
        def __init__(self, orbff, device=None):
            calls["calculator"] = (orbff, device)

    calculator = types.ModuleType("orb_models.forcefield.calculator")
    calculator.ORBCalculator = ORBCalculator
    forcefield = types.ModuleType("orb_models.forcefield")
    forcefield.pretrained = pretrained
    orb_models = types.ModuleType("orb_models")

    for name, module in {
        "torch": torch,
        "orb_models": orb_models,
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.pretrained": pretrained,
        "orb_models.forcefield.calculator": calculator,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(params=[_NVIDIA, _AMD], ids=["nvidia", "amd"])
def orb_env(request, monkeypatch):
    calls: dict = {}
    _install_fake_orb_models(monkeypatch, calls)
    return _load_env_module(request.param, "orb"), calls


def test_orb_hook_loads_file_via_arch_loader(orb_env):
    module, calls = orb_env
    module.setup_from_path("/scratch/me/ft.ckpt", device="cpu")
    assert calls["weights_path"] == "/scratch/me/ft.ckpt"
    assert calls["load_device"] == "device(cpu)"
    assert calls["calculator"] == ("orbff", "device(cpu)")


def test_orb_hook_default_arch_is_orb_v2(orb_env):
    module, calls = orb_env
    module.setup_from_path("/scratch/me/ft.ckpt", device="cpu", arch="orb-v2")
    default_calls = dict(calls)
    module.setup_from_path("/scratch/me/ft.ckpt", device="cpu")
    assert calls == default_calls


def test_orb_hook_rejects_unknown_arch(orb_env):
    module, _ = orb_env
    with pytest.raises(ValueError, match="unknown orb architecture"):
        module.setup_from_path("/scratch/me/ft.ckpt", device="cpu", arch="not-a-model")
