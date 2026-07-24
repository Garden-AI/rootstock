"""Sequentially pre-read env files into the OS page cache before heavy imports.

Python's import machinery mmaps large shared libraries and faults them in
roughly a page per synchronous read. On a cold page cache over a network
filesystem each fault is a full round-trip — measured on NCSA Delta's
HDD-backed Lustre at ~0.35 MB/s (hours for a torch-sized env) while the
same files streamed sequentially at 13 MB/s cold on the same node (#167).
Reading the bytes sequentially and discarding them populates the page
cache, so the mmap faults that follow hit memory instead of the wire.
When the cache is already warm this costs seconds of memory-speed reads.

``spawn_in_env`` stages this module next to the worker wrapper, which runs
it inside the *target env's* Python before any heavy import — so it must
stay stdlib-only and compatible with any Python an env might ship.

Files are read concurrently (Lustre stripes across OSTs, so a few
parallel streams multiply throughput), largest-first so a multi-GB
library starting late can't extend the tail single-threaded.
``ROOTSTOCK_PREWARM_THREADS`` overrides the reader count.

Best-effort by design: unreadable files are skipped, any unexpected error
aborts the prewarm and never the worker, and ``ROOTSTOCK_NO_PREWARM=1``
skips it entirely (e.g. on node-local or known-warm installs where it is
pure overhead).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_CHUNK_SIZE = 16 * 1024 * 1024
_DEFAULT_THREADS = 8


def _thread_count() -> int:
    try:
        n = int(os.environ.get("ROOTSTOCK_PREWARM_THREADS", ""))
    except ValueError:
        return _DEFAULT_THREADS
    return max(1, n)


def iter_prewarm_files(spec: dict):
    """Yield the files worth warming for a worker spawn spec.

    - every file under the env tree (the ``.so`` libraries dominate the
      volume, but imports also open thousands of small ``.py``/``.pyc``
      files whose cold reads are just as latency-bound — covering them adds
      little data and removes the residual small-file tail),
    - the local checkpoint weights file, when the spec names one,
    - any extra files or directory trees a client lists in
      ``spec["prewarm_paths"]`` (reserved for future use; directories are
      walked recursively).
    """
    env_dir = spec.get("env_dir")
    if env_dir:
        yield from sorted(p for p in Path(env_dir).rglob("*") if p.is_file())

    extras = list(spec.get("prewarm_paths") or [])
    if spec.get("checkpoint_path"):
        extras.append(spec["checkpoint_path"])
    for extra in extras:
        path = Path(extra)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            yield path


def _read_file(path) -> int:
    """Read one file front to back, discarding the bytes; returns its size."""
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            n += len(chunk)
    return n


def prewarm_files(paths, max_workers: int | None = None) -> tuple[int, int]:
    """Read ``paths`` concurrently, discarding the bytes.

    Each file is one unit of work (streamed sequentially within itself, so
    readahead still engages), dispatched largest-first across the reader
    pool. Duplicate paths are read once. Returns ``(n_files, n_bytes)``
    actually read. Files that vanish or can't be opened are skipped — the
    goal is a warm cache, not an audit.
    """
    # Size lookup doubles as dedup + existence filter; the rglob that
    # produced these paths just stat'ed them, so the attributes are still
    # in the client's metadata cache and this pass is cheap.
    sized: list[tuple[int, Path]] = []
    seen = set()
    for path in paths:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        try:
            sized.append((os.stat(path).st_size, path))
        except OSError:
            continue
    sized.sort(key=lambda item: item[0], reverse=True)

    if max_workers is None:
        max_workers = _thread_count()

    n_files = 0
    n_bytes = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_read_file, path) for _, path in sized]
        for future in futures:
            try:
                n_bytes += future.result()
            except OSError:
                continue
            n_files += 1
    return n_files, n_bytes


def prewarm_from_spec(spec: dict, log=None) -> None:
    """Warm the page cache for a worker spawn spec; never raises.

    The one-line summary goes to ``log`` (default stderr, so it lands in
    the worker's captured output and any post-mortem) — its duration is a
    direct read on filesystem health: seconds when warm, and when cold it
    replaces an hours-long stall with a visible, bounded read.
    """
    if os.environ.get("ROOTSTOCK_NO_PREWARM"):
        return
    if log is None:
        log = sys.stderr
    try:
        began = time.monotonic()
        n_files, n_bytes = prewarm_files(iter_prewarm_files(spec))
        elapsed = time.monotonic() - began
        print(
            f"[Worker] Prewarmed page cache: {n_files} files, "
            f"{n_bytes / 1e6:.0f} MB in {elapsed:.1f}s",
            file=log,
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - never take the worker down
        try:
            print(f"[Worker] Prewarm skipped: {type(exc).__name__}: {exc}", file=log, flush=True)
        except Exception:
            pass
