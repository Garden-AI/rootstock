"""VERIFY_KWARGS: parsing, lookup, and the verify_checkpoint fallback.

Envs whose ``setup()`` requires a kwarg (a task-head selection with no
default — UMA's ``task``, MACE-MH-1's ``head``) declare a module-level
``VERIFY_KWARGS`` so smoke-test and a bare ``rootstock add`` can still
verify. Explicit kwargs always win; the calculator path never reads it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rootstock import verify
from rootstock.environment import parse_verify_kwargs, verify_kwargs_for

DECLARING = """CHECKPOINTS = {"uma-s-1p1": "uma-s-1p1", "uma:custom": None}

VERIFY_KWARGS = {
    "uma-s-1p1": {"task": "omat"},
    "uma:custom": {"task": "omat"},
}

def setup(checkpoint, device="cuda", task=None):
    return None

def setup_from_path(path, device="cuda", task=None):
    return None
"""

PLAIN = """CHECKPOINTS = {"mace-mp-0-medium": "medium"}

def setup(checkpoint, device="cuda"):
    return None
"""


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "env.py"
    path.write_text(source)
    return path


def _install(root: Path, name: str, source: str) -> None:
    env_dir = root / "envs" / name
    env_dir.mkdir(parents=True)
    (env_dir / "env_source.py").write_text(source)


# --- parse_verify_kwargs -----------------------------------------------------


def test_parse_absent_means_empty(tmp_path):
    assert parse_verify_kwargs(_write(tmp_path, PLAIN)) == {}


def test_parse_declaration(tmp_path):
    parsed = parse_verify_kwargs(_write(tmp_path, DECLARING))
    assert parsed == {
        "uma-s-1p1": {"task": "omat"},
        "uma:custom": {"task": "omat"},
    }


def test_parse_annotated_assignment(tmp_path):
    source = 'VERIFY_KWARGS: dict = {"a": {"task": "omat"}}\n'
    assert parse_verify_kwargs(_write(tmp_path, source)) == {"a": {"task": "omat"}}


def test_parse_rejects_non_dict(tmp_path):
    with pytest.raises(ValueError, match="dict literal"):
        parse_verify_kwargs(_write(tmp_path, "VERIFY_KWARGS = ['nope']\n"))


def test_parse_rejects_non_literal(tmp_path):
    source = 'X = {"task": "omat"}\nVERIFY_KWARGS = {"a": X}\n'
    with pytest.raises(ValueError, match="dict literal"):
        parse_verify_kwargs(_write(tmp_path, source))


def test_parse_rejects_non_dict_values(tmp_path):
    with pytest.raises(ValueError, match="setup kwargs"):
        parse_verify_kwargs(_write(tmp_path, 'VERIFY_KWARGS = {"a": "omat"}\n'))


# --- verify_kwargs_for -------------------------------------------------------


def test_lookup_declared_entry(tmp_path):
    _install(tmp_path, "uma", DECLARING)
    assert verify_kwargs_for(tmp_path, "uma", "uma-s-1p1") == {"task": "omat"}
    assert verify_kwargs_for(tmp_path, "uma", "uma:custom") == {"task": "omat"}


def test_lookup_undeclared_checkpoint_is_empty(tmp_path):
    _install(tmp_path, "uma", DECLARING)
    assert verify_kwargs_for(tmp_path, "uma", "uma-m-1p1") == {}


def test_lookup_missing_source_is_empty(tmp_path):
    assert verify_kwargs_for(tmp_path, "ghost", "uma-s-1p1") == {}


# --- verify_checkpoint fallback ----------------------------------------------


@pytest.fixture
def capturing_server(monkeypatch):
    """Stub RootstockServer that records ctor kwargs and verifies cleanly."""
    captured: list[dict] = []

    class _Server:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def start(self):
            pass

        def calculate(self, **_):
            forces = np.array([[0.1, -0.2, 0.0], [-0.05, 0.1, 0.0], [-0.05, 0.1, 0.0]])
            return -14.0, forces, np.zeros((3, 3))

        def stop(self):
            pass

    monkeypatch.setattr("rootstock.server.RootstockServer", _Server)
    return captured


def test_empty_kwargs_fall_back_to_declaration(tmp_path, capturing_server):
    _install(tmp_path, "uma", DECLARING)
    ok, err = verify.verify_checkpoint(
        root=tmp_path, env_name="uma", checkpoint="uma-s-1p1", device="cpu", setup_kwargs={}
    )
    assert ok, err
    assert capturing_server[0]["setup_kwargs"] == {"task": "omat"}


def test_explicit_kwargs_win(tmp_path, capturing_server):
    _install(tmp_path, "uma", DECLARING)
    ok, err = verify.verify_checkpoint(
        root=tmp_path,
        env_name="uma",
        checkpoint="uma-s-1p1",
        device="cpu",
        setup_kwargs={"task": "omol"},
    )
    assert ok, err
    assert capturing_server[0]["setup_kwargs"] == {"task": "omol"}


def test_no_declaration_stays_empty(tmp_path, capturing_server):
    _install(tmp_path, "mace", PLAIN)
    ok, err = verify.verify_checkpoint(
        root=tmp_path, env_name="mace", checkpoint="mace-mp-0-medium", device="cpu"
    )
    assert ok, err
    assert capturing_server[0]["setup_kwargs"] == {}
