"""Atomic rebuilds: install builds into {root}/.build and swaps into envs/.

On a shared install, `install --force` used to delete the live env before the
(slow) build, so every worker spawn failed for the duration — and a failed
build left nothing at all. Now the live env serves traffic until the finished
build is renamed into place, and a failed build leaves it untouched.

``subprocess.run`` is mocked; the fake mimics uv's filesystem side effects
(creating the venv dir) so the real swap/cleanup logic is exercised.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

from rootstock.commands.install import _install_single_environment

ENV_SOURCE = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    '# dependencies = ["six>=1.0"]\n'
    "# ///\n"
    "CHECKPOINTS = {}\n"
    'def setup(checkpoint: str, device: str = "cuda"):\n'
    "    return None\n"
)


def _fake_uv(calls, fail_on=None):
    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        result = MagicMock()
        result.returncode = 1 if (fail_on and cmd[:2] == fail_on) else 0
        result.stderr = "boom" if result.returncode else ""
        result.stdout = ""
        if result.returncode == 0 and cmd[:2] == ["uv", "venv"]:
            venv_dir = Path(cmd[2])
            venv_dir.mkdir(parents=True, exist_ok=True)
            (venv_dir / "new-build-marker").touch()
        return result

    return fake_run


def _install(tmp_path, monkeypatch, calls, fake_run=None, force=False):
    monkeypatch.setattr("rootstock.__version__", "9.9.9")
    monkeypatch.setattr(
        "rootstock.commands.install.subprocess.run", fake_run or _fake_uv(calls)
    )
    monkeypatch.setattr(
        "rootstock.commands.install._precompile_environment", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "rootstock.commands.manifest.update_and_push_manifest", lambda *a, **k: None
    )
    env_file = tmp_path / "src" / "probe.py"
    env_file.parent.mkdir(exist_ok=True)
    env_file.write_text(ENV_SOURCE)
    return _install_single_environment(
        root=tmp_path, source=str(env_file), force=force, verbose=False
    )


def _make_live_env(tmp_path) -> Path:
    env_target = tmp_path / "envs" / "probe"
    env_target.mkdir(parents=True)
    (env_target / "live-env-marker").touch()
    return env_target


def test_fresh_install_lands_in_envs_and_cleans_build_dir(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    rc = _install(tmp_path, monkeypatch, calls)

    assert rc == 0
    assert (tmp_path / "envs" / "probe" / "new-build-marker").exists()
    assert list((tmp_path / ".build").iterdir()) == []


def test_venv_is_created_relocatable(tmp_path, monkeypatch):
    """The build happens in .build/ and is renamed into envs/, so script
    shebangs must not bake in the staging path."""
    calls: list[list[str]] = []

    rc = _install(tmp_path, monkeypatch, calls)

    assert rc == 0
    (venv_call,) = [c for c in calls if c[:2] == ["uv", "venv"]]
    assert "--relocatable" in venv_call
    assert venv_call[2] == str(tmp_path / ".build" / f"probe.{os.getpid()}")


def test_force_rebuild_swaps_env_without_predelete(tmp_path, monkeypatch):
    _make_live_env(tmp_path)
    calls: list[list[str]] = []

    rc = _install(tmp_path, monkeypatch, calls, force=True)

    assert rc == 0
    env_target = tmp_path / "envs" / "probe"
    assert (env_target / "new-build-marker").exists()
    assert not (env_target / "live-env-marker").exists()
    assert list((tmp_path / ".build").iterdir()) == []


def test_failed_build_leaves_live_env_untouched(tmp_path, monkeypatch):
    """The big one: a failed --force rebuild must not take down the env that
    was working before it started."""
    _make_live_env(tmp_path)
    calls: list[list[str]] = []

    rc = _install(
        tmp_path, monkeypatch, calls, fake_run=_fake_uv(calls, fail_on=["uv", "sync"]),
        force=True,
    )

    assert rc == 1
    env_target = tmp_path / "envs" / "probe"
    assert (env_target / "live-env-marker").exists()
    assert not (env_target / "new-build-marker").exists()
    assert list((tmp_path / ".build").iterdir()) == []


def test_stale_build_dirs_are_cleared(tmp_path, monkeypatch):
    """Leftovers from a crashed build of the same env don't accumulate."""
    stale = tmp_path / ".build" / "probe.99999"
    stale.mkdir(parents=True)
    (stale / "junk").touch()
    calls: list[list[str]] = []

    rc = _install(tmp_path, monkeypatch, calls)

    assert rc == 0
    assert not stale.exists()
