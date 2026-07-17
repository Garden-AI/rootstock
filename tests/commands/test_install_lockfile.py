"""Tests for the per-env lockfile step of ``rootstock install``.

A build is only as reproducible as its resolution, so install must:
1. resolve a lockfile with ``uv lock --script`` before syncing deps,
2. honor an existing lockfile by default and re-resolve only on --upgrade,
3. store the lockfile inside the built env (``env_source.py.lock``),
4. carry an authoritative lockfile shipped next to the source file.

``subprocess.run`` is mocked at the module boundary; the fake creates the
side effects each uv command would (venv dir, lockfile) so the surrounding
copy logic runs for real.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from rootstock.operations import _lockfile_for, install_environment

ENV_WITH_DEPS = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    '# dependencies = ["six>=1.0"]\n'
    "# ///\n"
    "CHECKPOINTS = {}\n"
    'def setup(checkpoint: str, device: str = "cuda"):\n'
    "    return None\n"
)

ENV_WITHOUT_DEPS = (
    "# /// script\n"
    '# requires-python = ">=3.10"\n'
    "# dependencies = []\n"
    "# ///\n"
    "CHECKPOINTS = {}\n"
    'def setup(checkpoint: str, device: str = "cuda"):\n'
    "    return None\n"
)

FAKE_LOCK = 'version = 1\nrequires-python = ">=3.10"\n'


def _fake_uv(captured_calls, lock_content=FAKE_LOCK, fail_on=None):
    """A subprocess.run stand-in that mimics uv's filesystem side effects."""

    def fake_run(cmd, **kwargs):
        captured_calls.append(list(cmd))
        result = MagicMock()
        result.returncode = 1 if (fail_on and cmd[:2] == fail_on) else 0
        result.stderr = "boom" if result.returncode else ""
        result.stdout = ""
        if result.returncode == 0:
            if cmd[:2] == ["uv", "venv"]:
                Path(cmd[2]).mkdir(parents=True, exist_ok=True)
            elif cmd[:2] == ["uv", "lock"]:
                # Like real uv: an existing lockfile's pins are kept (no
                # rewrite unless --upgrade); a missing one is created.
                script = Path(cmd[cmd.index("--script") + 1])
                lock = _lockfile_for(script)
                if "--upgrade" in cmd or not lock.exists():
                    lock.write_text(lock_content)
        return result

    return fake_run


def _install(tmp_path, monkeypatch, source: Path, calls, fake_run=None, **kwargs):
    monkeypatch.setattr("rootstock.__version__", "9.9.9")
    monkeypatch.setattr(
        "rootstock.operations.subprocess.run",
        fake_run or _fake_uv(calls),
    )
    monkeypatch.setattr("rootstock.operations._precompile_environment", lambda *a, **k: None)
    monkeypatch.setattr(
        "rootstock.operations.update_and_push_manifest",
        lambda *a, **k: None,
    )
    return install_environment(
        root=tmp_path,
        source=str(source),
        force=kwargs.pop("force", False),
        verbose=False,
        progress=print,
        **kwargs,
    )


def _write_env(tmp_path: Path, name: str = "probe", content: str = ENV_WITH_DEPS) -> Path:
    src_dir = tmp_path / "src"
    src_dir.mkdir(exist_ok=True)
    env_file = src_dir / f"{name}.py"
    env_file.write_text(content)
    return env_file


def test_lock_resolved_before_sync(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    env_file = _write_env(tmp_path)

    _install(tmp_path, monkeypatch, env_file, calls)

    canonical = tmp_path / "environments" / "probe.py"
    lock_calls = [c for c in calls if c[:2] == ["uv", "lock"]]
    sync_calls = [c for c in calls if c[:2] == ["uv", "sync"]]
    assert lock_calls == [["uv", "lock", "--script", str(canonical)]]
    assert sync_calls == [["uv", "sync", "--script", str(canonical), "--active", "--frozen"]]
    assert calls.index(lock_calls[0]) < calls.index(sync_calls[0])


def test_upgrade_flag_re_resolves(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    env_file = _write_env(tmp_path)

    _install(tmp_path, monkeypatch, env_file, calls, upgrade=True)

    canonical = tmp_path / "environments" / "probe.py"
    lock_calls = [c for c in calls if c[:2] == ["uv", "lock"]]
    assert lock_calls == [["uv", "lock", "--script", str(canonical), "--upgrade"]]


def test_lockfile_stored_in_built_env(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    env_file = _write_env(tmp_path)

    _install(tmp_path, monkeypatch, env_file, calls)

    stored = tmp_path / "envs" / "probe" / "env_source.py.lock"
    assert stored.read_text() == FAKE_LOCK


def test_existing_lockfile_honored_on_rebuild(tmp_path, monkeypatch, capsys):
    """Name-mode rebuild with a lockfile present must not pass --upgrade."""
    calls: list[list[str]] = []
    canonical = tmp_path / "environments" / "probe.py"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(ENV_WITH_DEPS)
    _lockfile_for(canonical).write_text(FAKE_LOCK)

    _install(tmp_path, monkeypatch, Path("probe"), calls)

    assert "Honoring existing lockfile" in capsys.readouterr().out
    lock_calls = [c for c in calls if c[:2] == ["uv", "lock"]]
    assert lock_calls == [["uv", "lock", "--script", str(canonical)]]


def test_source_adjacent_lockfile_registered(tmp_path, monkeypatch):
    """A lockfile shipped next to the source file rides along to environments/
    and is what the build (and the built env's stored copy) resolves from."""
    calls: list[list[str]] = []
    env_file = _write_env(tmp_path)
    _lockfile_for(env_file).write_text("carried = true\n")

    _install(tmp_path, monkeypatch, env_file, calls)

    canonical_lock = _lockfile_for(tmp_path / "environments" / "probe.py")
    assert canonical_lock.read_text() == "carried = true\n"
    stored = tmp_path / "envs" / "probe" / "env_source.py.lock"
    assert stored.read_text() == "carried = true\n"


def test_unlockable_env_builds_without_lockfile(tmp_path, monkeypatch, capsys):
    """`uv lock` resolves universally, so envs pulling wheels from a
    platform-specific find-links index (the PyG stacks) cannot be locked.
    That must not fail the build — sync falls back to a plain
    current-platform resolution, exactly the pre-lockfile behavior."""
    calls: list[list[str]] = []
    env_file = _write_env(tmp_path)

    _install(
        tmp_path,
        monkeypatch,
        env_file,
        calls,
        fake_run=_fake_uv(calls, fail_on=["uv", "lock"]),
    )

    assert "could not resolve a lockfile" in capsys.readouterr().err
    (sync_call,) = [c for c in calls if c[:2] == ["uv", "sync"]]
    assert "--frozen" not in sync_call
    assert not (tmp_path / "envs" / "probe" / "env_source.py.lock").exists()


def test_lock_failure_never_freezes_to_a_possibly_stale_lockfile(tmp_path, monkeypatch, capsys):
    """If re-locking fails while a lockfile exists, sync must NOT be frozen
    to it: the source may have drifted from those pins (e.g. a newly added
    dep that doesn't resolve), and --frozen would silently build the old env
    while claiming success. Sync validates the lock itself — a still-valid
    lock is used, a genuinely broken dependency fails there, loudly."""
    calls: list[list[str]] = []
    canonical = tmp_path / "environments" / "probe.py"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(ENV_WITH_DEPS)
    _lockfile_for(canonical).write_text(FAKE_LOCK)

    _install(
        tmp_path,
        monkeypatch,
        Path("probe"),
        calls,
        fake_run=_fake_uv(calls, fail_on=["uv", "lock"]),
    )

    assert "could not resolve a lockfile" in capsys.readouterr().err
    (sync_call,) = [c for c in calls if c[:2] == ["uv", "sync"]]
    assert "--frozen" not in sync_call


def test_no_dependencies_skips_lock(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    env_file = _write_env(tmp_path, name="noop", content=ENV_WITHOUT_DEPS)

    _install(tmp_path, monkeypatch, env_file, calls)

    assert not [c for c in calls if c[:2] == ["uv", "lock"]]
    assert not (tmp_path / "envs" / "noop" / "env_source.py.lock").exists()
