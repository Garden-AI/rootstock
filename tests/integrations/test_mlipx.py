"""Tests for the MLIPx integration node (rootstock.mlipx.RootstockMLIPxModel)."""

from __future__ import annotations

import pytest

# Skip the whole module cleanly if the optional 'mlipx' extra isn't installed.
pytest.importorskip("zntrack")

from rootstock.environment import CheckpointNotFoundError
from rootstock.integrations.mlipx import RootstockMLIPxModel


def test_get_spec_shape_and_metadata():
    model = RootstockMLIPxModel(checkpoint="mace-mp-0-medium", root="/tmp/no_env", device="cpu")
    spec = model.get_spec()
    assert set(spec) == {"metadata", "data"}
    assert spec["metadata"]["checkpoint"] == "mace-mp-0-medium"
    assert spec["metadata"]["backend"] == "rootstock"
    assert spec["metadata"]["device"] == "cpu"


def test_get_calculator_reaches_env_boundary():
    # Without an installed env, construction gets all the way to the checkpoint
    # lookup and raises. This proves the wiring reaches Rootstock without
    # requiring a built environment (so it runs in CI).
    model = RootstockMLIPxModel(
        checkpoint="mace-mp-0-medium", root="/tmp/no_env_here", device="cpu"
    )
    with pytest.raises(CheckpointNotFoundError):
        model.get_calculator()


def test_run_rejects_both_cluster_and_root():
    model = RootstockMLIPxModel(
        checkpoint="mace-mp-0-medium", cluster="sophia", root="/tmp/x", device="cpu"
    )
    with pytest.raises(ValueError):
        model.run()
