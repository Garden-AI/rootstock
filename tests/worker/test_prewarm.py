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
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rootstock.prewarm import (
    _cgroup_memory_limit,
    iter_prewarm_files,
    prewarm_files,
    prewarm_from_spec,
)
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


def test_summary_tags_weights_tier(fake_env, tmp_path: Path):
    """The tier tag is the field data for retiring the heuristic (#178)."""
    weights = tmp_path / "recorded.bin"
    weights.write_bytes(b"w" * 2_000_000)
    del fake_env["checkpoint_path"]
    fake_env["prewarm_paths"] = [str(weights)]
    fake_env["prewarm_weights_tier"] = "manifest"
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    assert "; weights: 2 MB (manifest)" in log.getvalue()


def test_summary_reports_none_recorded(fake_env):
    del fake_env["checkpoint_path"]
    fake_env["prewarm_paths"] = []
    fake_env["prewarm_weights_tier"] = "none"
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    assert "; weights: none recorded" in log.getvalue()


def test_summary_tags_custom_weights(fake_env):
    # fake_env carries a :custom checkpoint_path and no tier annotation.
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    assert "; weights: 0 MB (custom)" in log.getvalue()


def test_summary_has_no_weights_note_without_any(fake_env):
    del fake_env["checkpoint_path"]
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    assert "weights" not in log.getvalue()


def test_working_set_warning_when_cgroup_limit_too_small(fake_env, monkeypatch: pytest.MonkeyPatch):
    """Warming more than the job's memory cgroup holds evicts the warmed
    pages before the worker reads them (Delta, 2026-07-29) — one line names
    the footgun up front, before the reads start."""
    monkeypatch.setattr("rootstock.prewarm._cgroup_memory_limit", lambda: 100)
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    message = log.getvalue()
    assert "exceeds this job's memory limit" in message
    # The warning precedes the summary so it survives a stalled warm-up.
    assert message.index("Warning") < message.index("Prewarmed")


def test_no_working_set_warning_within_limit(fake_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("rootstock.prewarm._cgroup_memory_limit", lambda: 1 << 40)
    log = io.StringIO()
    prewarm_from_spec(fake_env, log=log)
    assert "Warning" not in log.getvalue()


def _cgroup_fixture(tmp_path: Path, proc_line: str) -> tuple[str, str]:
    proc = tmp_path / "proc_cgroup"
    proc.write_text(proc_line + "\n")
    sys_root = tmp_path / "sys_cgroup"
    sys_root.mkdir()
    return str(proc), str(sys_root)


def test_cgroup_v2_limit_found_on_ancestor(tmp_path: Path):
    """Schedulers set the limit on the job slice, not the leaf — the walk up
    must find it, and the leaf's 'max' placeholder must not mask it."""
    proc, sys_root = _cgroup_fixture(tmp_path, "0::/slurm/job_7/step_0")
    leaf = Path(sys_root) / "slurm" / "job_7" / "step_0"
    leaf.mkdir(parents=True)
    (leaf / "memory.max").write_text("max\n")
    (leaf.parent / "memory.max").write_text(str(32 * 1024**3) + "\n")
    assert _cgroup_memory_limit(proc, sys_root) == 32 * 1024**3


def test_cgroup_v1_limit(tmp_path: Path):
    proc, sys_root = _cgroup_fixture(tmp_path, "4:memory:/slurm/job_9")
    leaf = Path(sys_root) / "memory" / "slurm" / "job_9"
    leaf.mkdir(parents=True)
    (leaf / "memory.limit_in_bytes").write_text(str(16 * 1024**3) + "\n")
    assert _cgroup_memory_limit(proc, sys_root) == 16 * 1024**3


def test_cgroup_v1_no_limit_sentinel_ignored(tmp_path: Path):
    proc, sys_root = _cgroup_fixture(tmp_path, "4:memory:/user")
    leaf = Path(sys_root) / "memory" / "user"
    leaf.mkdir(parents=True)
    (leaf / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert _cgroup_memory_limit(proc, sys_root) is None


def test_cgroup_absent_means_no_limit(tmp_path: Path):
    assert _cgroup_memory_limit(str(tmp_path / "nope"), str(tmp_path)) is None


@pytest.fixture
def built_root(tmp_path: Path) -> Path:
    """A minimal built env so spawn_in_env accepts the spawn."""
    env_dir = tmp_path / "envs" / "fake"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()
    return tmp_path


def _sidecar(spec) -> dict:
    return json.loads(Path(spec.cmd[2]).read_text())


def test_spawn_fills_prewarm_paths_from_manifest(built_root: Path):
    (built_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "environments": {
                    "fake": {
                        "checkpoints": {
                            "fake-ckpt": {"weight_files": [{"path": "cache/w.pt", "size": 1}]}
                        }
                    }
                },
            }
        )
    )
    with spawn_in_env(built_root, "fake", WORKER_WRAPPER, {"checkpoint": "fake-ckpt"}) as spec:
        sidecar = _sidecar(spec)
    assert sidecar["prewarm_paths"] == [str(built_root / "cache" / "w.pt")]
    assert sidecar["prewarm_weights_tier"] == "manifest"


def test_spawn_reports_none_tier_without_record(built_root: Path):
    with spawn_in_env(built_root, "fake", WORKER_WRAPPER, {"checkpoint": "fake-ckpt"}) as spec:
        sidecar = _sidecar(spec)
    assert sidecar["prewarm_paths"] == []
    assert sidecar["prewarm_weights_tier"] == "none"


def test_spawn_keeps_caller_supplied_prewarm_paths(built_root: Path):
    payload = {"checkpoint": "fake-ckpt", "prewarm_paths": ["/explicit"]}
    with spawn_in_env(built_root, "fake", WORKER_WRAPPER, payload) as spec:
        sidecar = _sidecar(spec)
    assert sidecar["prewarm_paths"] == ["/explicit"]
    assert "prewarm_weights_tier" not in sidecar


def test_spawn_lookup_failure_is_not_fatal(built_root: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*args, **kwargs):
        raise RuntimeError("lookup exploded")

    monkeypatch.setattr("rootstock.spawn.get_checkpoint_prewarm_paths", boom)
    with spawn_in_env(built_root, "fake", WORKER_WRAPPER, {"checkpoint": "fake-ckpt"}) as spec:
        sidecar = _sidecar(spec)
    assert "prewarm_paths" not in sidecar


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
