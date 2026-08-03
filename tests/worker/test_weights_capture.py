"""Weight-file capture at add/verify time (issue #177).

The staged ``weights_capture`` module observes which files under the shared
cache trees a checkpoint's ``setup()`` actually touches — an audit hook for
Python-side opens plus a /proc/self/maps scan for mmap'd reads — and writes
them to a result file the parent turns into a manifest record.

Correctness here means: it records the right files (relative to the cache
root, deduped through symlinks, out-of-scope and empty files excluded), it
is a strict no-op unless the spec asks for capture, and it can never take
the download or the worker down. ``sys.addaudithook`` is process-global and
irreversible, so every test that installs the hook runs in a subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rootstock import weights_capture
from rootstock.spawn import DOWNLOAD_WRAPPER, WORKER_WRAPPER, spawn_in_env

_PKG_DIR = Path(weights_capture.__file__).parent


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    """A cache root with weights in both trees, plus decoys.

    ``cache/`` is where well-behaved libraries land (XDG_CACHE_HOME);
    ``home/`` catches the ones that hardcode ``~`` (fairchem). ``elsewhere/``
    must never be recorded, nor must zero-byte lock files.
    """
    root = tmp_path / "shared"
    (root / "cache" / "fake").mkdir(parents=True)
    (root / "home" / ".cache" / "fairchem").mkdir(parents=True)
    (root / "cache" / "fake" / "model.bin").write_bytes(b"w" * 4096)
    (root / "home" / ".cache" / "fairchem" / "uma.pt").write_bytes(b"u" * 2048)
    (root / "cache" / "fake" / "model.lock").write_bytes(b"")  # zero-size
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "outside.bin").write_bytes(b"o" * 1024)
    return root


def _spec(cache_root: Path, result_path: Path) -> dict:
    return {
        "weights_capture": {
            "result_path": str(result_path),
            "cache_root": str(cache_root),
        }
    }


def _run_driver(body: str, spec: dict) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter with the module imported the way
    the staged wrapper imports it (top-level, from a plain directory)."""
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(_PKG_DIR)!r})\n"
        "import weights_capture\n"
        "spec = json.loads(sys.argv[1])\n" + body
    )
    return subprocess.run(
        [sys.executable, "-c", script, json.dumps(spec)],
        capture_output=True,
        text=True,
    )


def _result(result_path: Path) -> dict:
    return json.loads(result_path.read_text())


# ---- the probes -------------------------------------------------------------


def test_capture_records_opened_files_relative_to_cache_root(cache_root, tmp_path):
    result_path = tmp_path / "weights.json"
    proc = _run_driver(
        "weights_capture.begin(spec)\n"
        f"open({str(cache_root / 'cache' / 'fake' / 'model.bin')!r}, 'rb').read()\n"
        f"open({str(cache_root / 'home' / '.cache' / 'fairchem' / 'uma.pt')!r}, 'rb').read()\n"
        f"open({str(cache_root / 'cache' / 'fake' / 'model.lock')!r}, 'rb').read()\n"
        f"open({str(tmp_path / 'elsewhere' / 'outside.bin')!r}, 'rb').read()\n"
        "weights_capture.finalize(spec)\n",
        _spec(cache_root, result_path),
    )
    assert proc.returncode == 0, proc.stderr

    files = {f["path"]: f["size"] for f in _result(result_path)["files"]}
    assert files == {
        "cache/fake/model.bin": 4096,  # relative to cache_root, so records
        "home/.cache/fairchem/uma.pt": 2048,  # survive split-cache installs
    }
    assert "Weight capture: 2 files" in proc.stderr


def test_opens_before_finalize_only(cache_root, tmp_path):
    """The hook goes dormant at finalize — a long-lived worker's later opens
    (an MD run's trajectory writes, say) never grow the record."""
    result_path = tmp_path / "weights.json"
    proc = _run_driver(
        "weights_capture.begin(spec)\n"
        "weights_capture.finalize(spec)\n"
        f"open({str(cache_root / 'cache' / 'fake' / 'model.bin')!r}, 'rb').read()\n",
        _spec(cache_root, result_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert _result(result_path)["files"] == []


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/maps is Linux-only")
def test_maps_scan_catches_mmap_reads_the_hook_missed(cache_root, tmp_path):
    """safetensors' native reader mmaps without a Python-visible open; the
    maps scan is the probe that catches it (the fd here is opened before the
    hook exists, so only the scan can see the file)."""
    result_path = tmp_path / "weights.json"
    proc = _run_driver(
        "import mmap\n"
        f"f = open({str(cache_root / 'cache' / 'fake' / 'model.bin')!r}, 'rb')\n"
        "weights_capture.begin(spec)\n"
        "m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)\n"
        "weights_capture.finalize(spec)\n"
        "m.close(); f.close()\n",
        _spec(cache_root, result_path),
    )
    assert proc.returncode == 0, proc.stderr
    files = [f["path"] for f in _result(result_path)["files"]]
    assert files == ["cache/fake/model.bin"]


def test_symlinked_file_records_once_as_its_real_path(cache_root, tmp_path):
    """HF hub's snapshots/ tree symlinks into blobs/; the physical file is
    what prewarm will read, and seeing both spellings must not double-count."""
    blob = cache_root / "cache" / "fake" / "model.bin"
    link = cache_root / "cache" / "fake" / "snapshot.bin"
    link.symlink_to(blob)
    result_path = tmp_path / "weights.json"
    proc = _run_driver(
        "weights_capture.begin(spec)\n"
        f"open({str(link)!r}, 'rb').read()\n"
        f"open({str(blob)!r}, 'rb').read()\n"
        "weights_capture.finalize(spec)\n",
        _spec(cache_root, result_path),
    )
    assert proc.returncode == 0, proc.stderr
    files = [f["path"] for f in _result(result_path)["files"]]
    assert files == ["cache/fake/model.bin"]


def test_parse_maps_filters_scope_anonymous_and_deleted():
    scopes = ("/shared/cache/", "/shared/home/")
    maps = "\n".join(
        [
            "7f0000-7f1000 r--p 00000000 08:01 1 /shared/cache/fake/model.bin",
            "7f2000-7f3000 rw-p 00000000 00:00 0",  # anonymous
            "7f4000-7f5000 r-xp 00000000 08:01 2 /usr/lib/libc.so.6",
            "7f6000-7f7000 r--p 00000000 08:01 3 /shared/home/w.pt (deleted)",
            "7f8000-7f9000 r--p 00000000 08:01 4 /shared/home/name with spaces.pt",
        ]
    )
    assert weights_capture.parse_maps(maps, scopes) == {
        "/shared/cache/fake/model.bin",
        "/shared/home/name with spaces.pt",
    }


# ---- no-op and never-fatal guarantees ---------------------------------------


def test_begin_and_wrap_setup_are_noops_without_spec_key():
    # Safe in-process: without the spec key no audit hook is installed.
    weights_capture.begin({})
    assert weights_capture._state is None

    def setup(checkpoint, device):
        return "calc"

    assert weights_capture.wrap_setup(setup, {}) is setup


def test_wrap_setup_writes_record_the_moment_setup_returns(cache_root, tmp_path):
    result_path = tmp_path / "weights.json"
    proc = _run_driver(
        "weights_capture.begin(spec)\n"
        "def setup(checkpoint, device='cpu'):\n"
        f"    open({str(cache_root / 'cache' / 'fake' / 'model.bin')!r}, 'rb').read()\n"
        "    return 'calc'\n"
        "wrapped = weights_capture.wrap_setup(setup, spec)\n"
        "assert wrapped('ckpt', device='cpu') == 'calc'\n",
        _spec(cache_root, result_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert [f["path"] for f in _result(result_path)["files"]] == ["cache/fake/model.bin"]


def test_failed_setup_writes_no_record(cache_root, tmp_path):
    """A load that died has no working set worth recording — the parent must
    see 'no capture ran', not a half-truth."""
    result_path = tmp_path / "weights.json"
    proc = _run_driver(
        "weights_capture.begin(spec)\n"
        "def setup(checkpoint, device='cpu'):\n"
        "    raise RuntimeError('download exploded')\n"
        "wrapped = weights_capture.wrap_setup(setup, spec)\n"
        "try:\n"
        "    wrapped('ckpt')\n"
        "except RuntimeError:\n"
        "    pass\n",
        _spec(cache_root, result_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert not result_path.exists()


def test_capture_failure_is_never_fatal(cache_root, tmp_path):
    """An unwritable result path logs and moves on — capture must never take
    down the download or the worker."""
    proc = _run_driver(
        "weights_capture.begin(spec)\n"
        f"open({str(cache_root / 'cache' / 'fake' / 'model.bin')!r}, 'rb').read()\n"
        "weights_capture.finalize(spec)\n"
        "print('still alive')\n",
        _spec(cache_root, tmp_path / "no" / "such" / "dir" / "weights.json"),
    )
    assert proc.returncode == 0, proc.stderr
    assert "Weight capture failed" in proc.stderr
    assert "still alive" in proc.stdout


# ---- wrapper integration ----------------------------------------------------


def _fake_built_env(root: Path, setup_body: str) -> None:
    env_dir = root / "envs" / "fake"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    (env_dir / "env_source.py").write_text(
        f"def setup(checkpoint, device='cpu', **kwargs):\n{setup_body}"
    )


def test_download_wrapper_runs_capture_end_to_end(tmp_path: Path):
    """Through the real staging + subprocess path: setup() touches a file
    under the spawn-provided cache env, and the record lands at result_path
    after the process exits — exactly what fetch_checkpoint reads."""
    weights = tmp_path / "cache" / "fake"
    weights.mkdir(parents=True)
    (weights / "model.bin").write_bytes(b"w" * 4096)
    _fake_built_env(
        tmp_path,
        f"    open({str(weights / 'model.bin')!r}, 'rb').read()\n    return None\n",
    )

    result_path = tmp_path / "weights.json"
    payload = {
        "checkpoint": "c",
        "device": "cpu",
        "setup_kwargs": {},
        "weights_capture": {"result_path": str(result_path), "cache_root": str(tmp_path)},
    }
    with spawn_in_env(tmp_path, "fake", DOWNLOAD_WRAPPER, payload) as spec:
        result = subprocess.run(
            spec.cmd, env=spec.env, cwd=spec.cwd, capture_output=True, text=True
        )
    assert result.returncode == 0, result.stderr
    assert "Weight capture: 1 files" in result.stderr
    files = json.loads(result_path.read_text())["files"]
    assert files == [{"path": "cache/fake/model.bin", "size": 4096}]


def test_download_wrapper_without_capture_key_stays_silent(tmp_path: Path):
    """No spec key, no capture — today's spawns are byte-for-byte unaffected."""
    _fake_built_env(tmp_path, "    return None\n")

    payload = {"checkpoint": "c", "device": "cpu", "setup_kwargs": {}}
    with spawn_in_env(tmp_path, "fake", DOWNLOAD_WRAPPER, payload) as spec:
        result = subprocess.run(
            spec.cmd, env=spec.env, cwd=spec.cwd, capture_output=True, text=True
        )
    assert result.returncode == 0, result.stderr
    assert "Weight capture" not in result.stderr


def test_spawn_stages_weights_capture_module_next_to_wrapper(tmp_path: Path):
    """The staged copy IS the implementation the env's python runs — that's
    what lets capture reach already-deployed envs without a rebuild."""
    env_dir = tmp_path / "envs" / "fake"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").touch()

    with spawn_in_env(tmp_path, "fake", WORKER_WRAPPER, {"checkpoint": "x"}) as spec:
        staged = Path(spec.cmd[1]).parent / "weights_capture.py"
        assert staged.exists()
        assert staged.read_text() == (_PKG_DIR / "weights_capture.py").read_text()
