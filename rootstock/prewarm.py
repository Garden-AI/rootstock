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

Best-effort by design: unreadable files are skipped, any unexpected error
aborts the prewarm and never the worker, and ``ROOTSTOCK_NO_PREWARM=1``
skips it entirely (e.g. on node-local or known-warm installs where it is
pure overhead).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_CHUNK_SIZE = 16 * 1024 * 1024


def iter_prewarm_files(spec: dict):
    """Yield the files worth warming for a worker spawn spec.

    - every shared library under the env (``{env_dir}/**/*.so*`` — the bulk
      of the import cost; torch dominates),
    - the local checkpoint weights file, when the spec names one,
    - any extra files or directory trees a client lists in
      ``spec["prewarm_paths"]`` (reserved for future use; directories are
      walked recursively).
    """
    env_dir = spec.get("env_dir")
    if env_dir:
        yield from sorted(Path(env_dir).rglob("*.so*"))

    extras = list(spec.get("prewarm_paths") or [])
    if spec.get("checkpoint_path"):
        extras.append(spec["checkpoint_path"])
    for extra in extras:
        path = Path(extra)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            yield path


def prewarm_files(paths) -> tuple[int, int]:
    """Sequentially read ``paths``, discarding the bytes.

    Returns ``(n_files, n_bytes)`` actually read. Files that vanish or
    can't be opened are skipped — the goal is a warm cache, not an audit.
    """
    n_files = 0
    n_bytes = 0
    for path in paths:
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    n_bytes += len(chunk)
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
