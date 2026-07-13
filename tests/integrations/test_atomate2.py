"""Tests for the atomate2 integration."""

import pytest

pytest.importorskip("atomate2")

from rootstock.integrations.atomate2 import (  # noqa: E402
    RootstockRelaxMaker,
    RootstockStaticMaker,
)

MAKERS = [RootstockRelaxMaker, RootstockStaticMaker]


@pytest.mark.parametrize("maker_cls", MAKERS)
def test_checkpoint_is_required(maker_cls, tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        maker_cls(root=tmp_path)


@pytest.mark.parametrize("maker_cls", MAKERS)
def test_cluster_and_root_are_mutually_exclusive(maker_cls, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        maker_cls(checkpoint="mace-mp-0-medium", cluster="sophia", root=tmp_path)

    with pytest.raises(ValueError, match="exactly one"):
        maker_cls(checkpoint="mace-mp-0-medium")


@pytest.mark.parametrize("maker_cls", MAKERS)
def test_calculator_meta_resolves_to_rootstock(maker_cls, tmp_path):
    """atomate2 must be able to import RootstockCalculator from the meta string.

    `ase_calculator_name` runs atomate2's `_load_calc_cls` on `calculator_meta`,
    so this fails loudly if the import path ever goes stale.
    """
    maker = maker_cls(checkpoint="mace-mp-0-medium", root=tmp_path)
    assert maker.calculator_meta == "rootstock.calculator.RootstockCalculator"
    assert maker.ase_calculator_name == "RootstockCalculator"


@pytest.mark.parametrize("maker_cls", MAKERS)
def test_device_defaults_to_cpu(maker_cls, tmp_path):
    """RootstockCalculator defaults to cuda; the Makers deliberately do not."""
    maker = maker_cls(checkpoint="mace-mp-0-medium", root=tmp_path)
    assert maker.device == "cpu"
    assert maker.close_worker is True


@pytest.mark.parametrize("maker_cls", MAKERS)
def test_close_is_a_noop_before_any_run(maker_cls, tmp_path):
    maker = maker_cls(checkpoint="mace-mp-0-medium", root=tmp_path)
    maker.close()  # no calculator built yet, must not raise