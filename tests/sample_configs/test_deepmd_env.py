"""Tests for the deepmd env's loading contract.

deepmd-kit resolves its built-in pretrained models by registry name and
fixes its PyTorch device at first import from the environment; the env file
routes both through Rootstock's conventions — weights into the shared cache
via XDG_CACHE_HOME, the device from the ``device`` argument, an explicit
``head`` for multitask checkpoints, and charge/spin read from the
``atoms.info`` keys the other OMol-trained envs use.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from ase import Atoms

from rootstock.environment import parse_verify_kwargs

_CONFIG = (
    Path(__file__).parent.parent.parent
    / "sample_model_configurations"
    / "nvidia_configs"
    / "deepmd.py"
)


def _load_env_module():
    spec = importlib.util.spec_from_file_location("deepmd_env", _CONFIG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDeepPot:
    def __init__(self, dim_fparam: int, chg_spin_ebd: bool):
        self._dim_fparam = dim_fparam
        self._chg_spin_ebd = chg_spin_ebd

    def get_dim_fparam(self) -> int:
        return self._dim_fparam

    def has_chg_spin_ebd(self) -> bool:
        return self._chg_spin_ebd


@pytest.fixture
def stubbed_deepmd(monkeypatch):
    """Stub deepmd in sys.modules; capture what setup() passes to each piece."""
    captured: dict = {"model_traits": (0, False)}

    class DP:
        def __init__(self, model, head=None, **kwargs):
            captured["calculator"] = {"model": model, "head": head, **kwargs}
            self.dp = _FakeDeepPot(*captured["model_traits"])
            self.results = {}

        def calculate(self, atoms=None, properties=None, system_changes=None):
            captured["calc_info"] = dict(atoms.info)

    def resolve_model_path(name, *, cache_dir=None):
        captured["resolve"] = {"name": name, "cache_dir": cache_dir}
        return Path(cache_dir) / f"{name}.pt"

    deepmd = types.ModuleType("deepmd")
    calculator = types.ModuleType("deepmd.calculator")
    calculator.DP = DP
    pretrained = types.ModuleType("deepmd.pretrained")
    download = types.ModuleType("deepmd.pretrained.download")
    download.resolve_model_path = resolve_model_path
    for name, module in {
        "deepmd": deepmd,
        "deepmd.calculator": calculator,
        "deepmd.pretrained": pretrained,
        "deepmd.pretrained.download": download,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setenv("XDG_CACHE_HOME", "/shared/root/cache")
    monkeypatch.delenv("DEVICE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    return captured


# ---------- registry resolution + head selection --------------------------------


def test_setup_resolves_registry_name_into_shared_cache(stubbed_deepmd):
    module = _load_env_module()
    module.setup("dpa-3.2-5m", device="cuda", head="OMat24")

    assert stubbed_deepmd["resolve"] == {
        "name": "DPA-3.2-5M",
        "cache_dir": Path("/shared/root/cache/deepmd/pretrained/models"),
    }
    assert stubbed_deepmd["calculator"] == {
        "model": "/shared/root/cache/deepmd/pretrained/models/DPA-3.2-5M.pt",
        "head": "OMat24",
    }


def test_multitask_checkpoint_requires_head(stubbed_deepmd):
    module = _load_env_module()
    with pytest.raises(ValueError, match="OMat24"):
        module.setup("dpa-3.2-5m", device="cuda")
    assert "resolve" not in stubbed_deepmd, "must fail before touching the cache"


def test_single_task_checkpoint_takes_no_head(stubbed_deepmd):
    module = _load_env_module()
    module.setup("dpa3-omol-large", device="cuda")
    assert stubbed_deepmd["calculator"]["head"] is None


def test_extra_kwargs_reach_the_calculator(stubbed_deepmd):
    module = _load_env_module()
    module.setup("dpa4-mini-omat24", device="cuda", nlist_backend="ase")
    assert stubbed_deepmd["calculator"]["nlist_backend"] == "ase"


def test_every_multitask_id_has_a_verify_head():
    module = _load_env_module()
    verify = parse_verify_kwargs(_CONFIG)
    for ckpt, heads in module.HEADS.items():
        assert ckpt in module.CHECKPOINTS
        head = verify[ckpt]["head"]
        assert head.lower() in {h.lower() for h in heads}
    for ckpt in module.CHECKPOINTS:
        if ckpt not in module.HEADS and ckpt != "dpa:custom":
            assert ckpt not in verify, f"{ckpt} is single-task and needs no verify head"


# ---------- device selection ----------------------------------------------------


def test_indexed_cuda_device_sets_local_rank(stubbed_deepmd, monkeypatch):
    module = _load_env_module()
    module.setup("dpa3-omol-large", device="cuda:2")
    import os

    assert os.environ["LOCAL_RANK"] == "2"
    assert "DEVICE" not in os.environ


def test_cpu_device_sets_deepmd_device_env(stubbed_deepmd):
    module = _load_env_module()
    module.setup("dpa3-omol-large", device="cpu")
    import os

    assert os.environ["DEVICE"] == "cpu"


# ---------- custom weights -------------------------------------------------------


def test_setup_from_path_loads_the_file_directly(stubbed_deepmd):
    module = _load_env_module()
    module.setup_from_path("/scratch/me/ft.pt", device="cpu", head="MyHead")
    assert "resolve" not in stubbed_deepmd
    assert stubbed_deepmd["calculator"] == {"model": "/scratch/me/ft.pt", "head": "MyHead"}


# ---------- charge/spin translation --------------------------------------------


def _water():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
    atoms.info["charge"] = 1
    atoms.info["spin"] = 2
    return atoms


def test_charge_spin_become_fparam_for_fparam_models(stubbed_deepmd):
    stubbed_deepmd["model_traits"] = (2, False)
    module = _load_env_module()
    calc = module.setup("dpa3-omol-large", device="cpu")
    atoms = _water()
    calc.calculate(atoms)

    assert stubbed_deepmd["calc_info"]["fparam"] == [1.0, 2.0]
    assert "charge_spin" not in stubbed_deepmd["calc_info"]
    assert "fparam" not in atoms.info, "the caller's atoms must not be mutated"


def test_charge_spin_become_charge_spin_for_embedding_models(stubbed_deepmd):
    stubbed_deepmd["model_traits"] = (0, True)
    module = _load_env_module()
    calc = module.setup("dpa-3.3-1m", device="cpu", head="OMat24")
    calc.calculate(_water())

    assert stubbed_deepmd["calc_info"]["charge_spin"] == [1.0, 2.0]
    assert "fparam" not in stubbed_deepmd["calc_info"]


def test_explicit_fparam_wins_over_charge_spin(stubbed_deepmd):
    stubbed_deepmd["model_traits"] = (2, False)
    module = _load_env_module()
    calc = module.setup("dpa3-omol-large", device="cpu")
    atoms = _water()
    atoms.info["fparam"] = [0.0, 1.0]
    calc.calculate(atoms)
    assert stubbed_deepmd["calc_info"]["fparam"] == [0.0, 1.0]


def test_models_without_charge_spin_inputs_are_left_alone(stubbed_deepmd):
    stubbed_deepmd["model_traits"] = (0, False)
    module = _load_env_module()
    calc = module.setup("dpa4-mini-omat24", device="cpu")
    calc.calculate(_water())
    assert "fparam" not in stubbed_deepmd["calc_info"]
    assert "charge_spin" not in stubbed_deepmd["calc_info"]


def test_atoms_without_charge_spin_pass_through_untouched(stubbed_deepmd):
    stubbed_deepmd["model_traits"] = (2, True)
    module = _load_env_module()
    calc = module.setup("dpa3-omol-large", device="cpu")
    calc.calculate(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    assert stubbed_deepmd["calc_info"] == {}
