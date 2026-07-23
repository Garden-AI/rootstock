"""Tests for ``rootstock add-local``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rootstock import local_checkpoints
from rootstock.commands import local as local_cmd
from rootstock.commands.local import cmd_add_local
from rootstock.local_checkpoints import local_checkpoints_for_root

_UMA_ENV_SOURCE = '''\
"""UMA env with the local-checkpoint hook."""

CHECKPOINTS = {
    "uma-s-1p1": "uma-s-1p1",
}


def setup(checkpoint, device="cuda", task="omat"):
    return None


def setup_from_path(path, device="cuda", task="omat"):
    return None
'''

_MACE_ENV_SOURCE = '''\
"""MACE env without the hook."""

CHECKPOINTS = {
    "mace-mp-0-medium": "medium",
}


def setup(checkpoint, device="cuda"):
    return None
'''


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for name, source in (("uma", _UMA_ENV_SOURCE), ("mace", _MACE_ENV_SOURCE)):
        env_dir = root / "envs" / name
        (env_dir / "bin").mkdir(parents=True)
        (env_dir / "bin" / "python").touch()
        (env_dir / "env_source.py").write_text(source)
    return root


@pytest.fixture
def registry(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "registry.json"
    monkeypatch.setattr(local_checkpoints, "DEFAULT_LOCAL_REGISTRY_FILE", path)
    return path


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "ft.pt"
    path.write_bytes(b"fine-tuned weights")
    return path


@pytest.fixture
def verify_calls(monkeypatch) -> list:
    """Stub verify_checkpoint where cmd_add_local binds it; record calls."""
    calls = []

    def fake_verify(root, env, checkpoint, device, **kwargs):
        calls.append({"env": env, "checkpoint": checkpoint, "device": device, **kwargs})
        return True, None

    monkeypatch.setattr(local_cmd, "verify_checkpoint", fake_verify)
    return calls


def _make_args(root: Path, weights: Path, **overrides):
    class _Args:
        pass

    args = _Args()
    args.path = str(overrides.get("path", weights))
    args.env = overrides.get("env", "uma")
    args.id = overrides.get("id", "my-uma-ft")
    args.kwarg = overrides.get("kwarg")
    args.device = overrides.get("device", "cuda")
    args.no_verify = overrides.get("no_verify", False)
    args.root = str(root)
    return args


def test_add_local_happy_path(fake_root, weights, registry, verify_calls):
    rc = cmd_add_local(_make_args(fake_root, weights, kwarg=["task=omol"]))
    assert rc == 0

    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert entry.env == "uma"
    assert entry.path == str(weights.resolve())
    assert entry.setup_kwargs == {"task": "omol"}
    assert entry.verified_at is not None
    assert entry.verified_device == "cuda"

    # Verification ran against the registered path with registered kwargs.
    assert verify_calls == [
        {
            "env": "uma",
            "checkpoint": "my-uma-ft",
            "device": "cuda",
            "setup_kwargs": {"task": "omol"},
            "cache_root": fake_root,
            "checkpoint_path": str(weights.resolve()),
        }
    ]


def test_add_local_no_verify(fake_root, weights, registry, verify_calls):
    rc = cmd_add_local(_make_args(fake_root, weights, no_verify=True))
    assert rc == 0
    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert entry.verified_at is None
    assert verify_calls == []


def test_add_local_verify_failure_keeps_entry(fake_root, weights, registry, monkeypatch):
    monkeypatch.setattr(
        local_cmd, "verify_checkpoint", lambda *a, **kw: (False, "CUDA out of memory")
    )
    rc = cmd_add_local(_make_args(fake_root, weights))
    assert rc == 1
    entry = local_checkpoints_for_root(fake_root)["my-uma-ft"]
    assert entry.verified_at is None
    assert entry.last_error == "verify: CUDA out of memory"


def test_add_local_env_without_hook(fake_root, weights, registry, verify_calls, capsys):
    rc = cmd_add_local(_make_args(fake_root, weights, env="mace"))
    assert rc == 1
    assert "setup_from_path" in capsys.readouterr().err
    assert local_checkpoints_for_root(fake_root) == {}


def test_add_local_canonical_collision(fake_root, weights, registry, verify_calls, capsys):
    rc = cmd_add_local(_make_args(fake_root, weights, id="uma-s-1p1"))
    assert rc == 1
    assert "canonical" in capsys.readouterr().err


def test_add_local_reserved_kwarg(fake_root, weights, registry, verify_calls, capsys):
    rc = cmd_add_local(_make_args(fake_root, weights, kwarg=["path=/x"]))
    assert rc == 1
    assert "reserved" in capsys.readouterr().err


def test_add_local_bad_kwarg_is_usage_error(fake_root, weights, registry, verify_calls):
    rc = cmd_add_local(_make_args(fake_root, weights, kwarg=["no-equals-sign"]))
    assert rc == 2


def test_add_local_missing_weights(fake_root, registry, verify_calls, capsys, tmp_path):
    rc = cmd_add_local(_make_args(fake_root, tmp_path / "nope.pt"))
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_add_local_leaves_umask_alone(fake_root, weights, registry, verify_calls):
    # cmd_add forces umask 0o002 for shared-cache writes; add-local writes
    # nothing shared and must not touch the caller's umask.
    before = os.umask(0o077)
    os.umask(before)
    cmd_add_local(_make_args(fake_root, weights))
    after = os.umask(0o077)
    os.umask(after)
    assert after == before
