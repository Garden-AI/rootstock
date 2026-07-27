"""Page-cache prewarm at worker spawn (issue #167).

On a cold page cache over Lustre, the worker's imports fault mmap'd
shared libraries in at network-round-trip speed (~0.35 MB/s measured on
Delta) while sequential reads of the same files run at bandwidth. The
prewarm module — staged next to the spawn wrapper, run inside the env's
Python before any heavy import — reads the env's ``.so`` files and the
checkpoint sequentially to populate the cache.

Correctness here means: it reads the right files, it is skippable, and it
can never take the worker down. The staging test pins the contract that
the module travels with the wrapper (that's what lets already-deployed
envs benefit without a rebuild).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rootstock.prewarm import iter_prewarm_files, prewarm_files, prewarm_from_spec
from rootstock.spawn import DOWNLOAD_WRAPPER, WORKER_WRAPPER, spawn_in_env


@pytest.fixture
def fake_env(tmp_path: Path) -> dict:
    """A spec pointing at an env tree with libraries, noise, and weights."""
    env_dir = tmp_path / "envs" / "fake"
    lib = env_dir / "lib" / "python3.11" / "site-packages" / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib / "libtorch_cpu.so").write_bytes(b"x" * 1024)
    (lib / "libc10.so.1").write_bytes(b"y" * 512)
    (env_dir / "env_source.py").write_text("# not a shared library")

    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"w" * 256)

    return {
        "env_dir": str(env_dir),
        "checkpoint": "fake-ckpt",
        "checkpoint_path": str(weights),
    }


def test_iter_finds_whole_env_tree_and_checkpoint(fake_env):
    # Whole tree, not just shared libraries: small .py/.pyc reads are just
    # as latency-bound on a cold cache (#170).
    found = {p.name for p in iter_prewarm_files(fake_env)}
    assert found == {"libtorch_cpu.so", "libc10.so.1", "env_source.py", "weights.pt"}


def test_iter_walks_extra_prewarm_dirs(fake_env, tmp_path: Path):
    cache = tmp_path / "cache" / "fake"
    cache.mkdir(parents=True)
    (cache / "model.bin").write_bytes(b"m")
    fake_env["prewarm_paths"] = [str(cache)]
    found = {p.name for p in iter_prewarm_files(fake_env)}
    assert "model.bin" in found


def test_prewarm_reads_every_byte(fake_env):
    expected_bytes = sum(p.stat().st_size for p in iter_prewarm_files(fake_env))
    n_files, n_bytes = prewarm_files(iter_prewarm_files(fake_env))
    assert n_files == 4
    assert n_bytes == expected_bytes


def test_single_reader_matches_parallel(fake_env):
    parallel = prewarm_files(iter_prewarm_files(fake_env))
    serial = prewarm_files(iter_prewarm_files(fake_env), max_workers=1)
    assert serial == parallel


def test_thread_knob_tolerates_garbage(fake_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROOTSTOCK_PREWARM_THREADS", "banana")
    n_files, _ = prewarm_files(iter_prewarm_files(fake_env))
    assert n_files == 4


def test_duplicate_paths_read_once(fake_env):
    # The checkpoint listed again via prewarm_paths must not double-count.
    fake_env["prewarm_paths"] = [fake_env["checkpoint_path"]]
    n_files, n_bytes = prewarm_files(iter_prewarm_files(fake_env))
    assert n_files == 4
    assert n_bytes == sum(p.stat().st_size for p in set(iter_prewarm_files(fake_env)))


def test_prewarm_from_spec_reports_summary(fake_env):
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    message = log.getvalue()
    assert "Prewarmed page cache: 4 files" in message


def test_env_var_skips_prewarm(fake_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ROOTSTOCK_NO_PREWARM", "1")
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    assert log.getvalue() == ""


def test_unreadable_file_is_skipped(fake_env):
    blocked = Path(fake_env["env_dir"]) / "lib" / "blocked.so"
    blocked.write_bytes(b"z" * 128)
    os.chmod(blocked, 0o000)
    try:
        n_files, n_bytes = prewarm_files(iter_prewarm_files(fake_env))
    finally:
        os.chmod(blocked, 0o644)  # let tmp_path cleanup work
    assert n_files == 4  # the readable four, blocked.so skipped
    assert n_bytes == 1024 + 512 + 256 + len("# not a shared library")


def test_prewarm_from_spec_never_raises():
    # Garbage spec: iter explodes on a non-dict-shaped value inside.
    log = io.StringIO()
    prewarm_from_spec({"env_dir": 0xBAD, "prewarm_paths": 42}, log=log)
    assert "Prewarm skipped" in log.getvalue()


def test_wrapper_runs_prewarm_end_to_end(tmp_path: Path):
    """Through the real staging + subprocess path, the summary line lands on
    the worker's stderr — which is what the post-mortem capture reads."""
    env_dir = tmp_path / "envs" / "fake"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "lib").mkdir()
    (env_dir / "lib" / "libfake.so").write_bytes(b"x" * 2048)
    (env_dir / "env_source.py").write_text(
        "def setup(checkpoint, device='cpu', **kwargs):\n    return None\n"
    )

    payload = {"checkpoint": "c", "device": "cpu", "setup_kwargs": {}}
    with spawn_in_env(tmp_path, "fake", DOWNLOAD_WRAPPER, payload) as spec:
        result = subprocess.run(
            spec.cmd, env=spec.env, cwd=spec.cwd, capture_output=True, text=True
        )
    assert result.returncode == 0, result.stderr
    # Whole tree: libfake.so + env_source.py + bin/python (the symlinked
    # interpreter binary — warming it is a feature, not an accident).
    assert "Prewarmed page cache: 3 files" in result.stderr


def test_spawn_stages_prewarm_module_next_to_wrapper(tmp_path: Path):
    env_dir = tmp_path / "envs" / "fake"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()

    with spawn_in_env(tmp_path, "fake", WORKER_WRAPPER, {"checkpoint": "x"}) as spec:
        staged_dir = Path(spec.cmd[1]).parent
        staged = staged_dir / "prewarm.py"
        assert staged.exists()
        # Byte-identical to the client's module: the staged copy IS the
        # implementation the env's python runs.
        source = Path(__file__).resolve().parents[2] / "rootstock" / "prewarm.py"
        assert staged.read_text() == source.read_text()
