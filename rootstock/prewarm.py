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


def _iter_env_files(spec: dict):
    """Every file under the env tree: the ``.so`` libraries dominate the
    volume, but imports also open thousands of small ``.py``/``.pyc`` files
    whose cold reads are just as latency-bound — covering them adds little
    data and removes the residual small-file tail."""
    env_dir = spec.get("env_dir")
    if env_dir:
        yield from sorted(p for p in Path(env_dir).rglob("*") if p.is_file())


def _iter_weight_files(spec: dict):
    """The spawn's model-weight files, outside the env tree:

    - checkpoint weight paths in ``spec["prewarm_paths"]``, filled by
      spawn_in_env from the checkpoint's manifest record or the cache-scan
      heuristic (#178; directories are walked recursively),
    - the user-supplied (:custom) weights file, when the spec names one.
    """
    extras = list(spec.get("prewarm_paths") or [])
    if spec.get("checkpoint_path"):
        extras.append(spec["checkpoint_path"])
    for extra in extras:
        path = Path(extra)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            yield path


def iter_prewarm_files(spec: dict):
    """Yield the files worth warming for a worker spawn spec: the whole env
    tree, then the checkpoint's weight files (see the helpers above)."""
    yield from _iter_env_files(spec)
    yield from _iter_weight_files(spec)


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


def _stat_files(paths) -> list[tuple[int, Path]]:
    """Dedup ``paths`` and pair each with its size, dropping unstattable ones.

    The size lookup doubles as dedup + existence filter; the rglob that
    produced these paths just stat'ed them, so the attributes are still
    in the client's metadata cache and this pass is cheap.
    """
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
    return sized


def _read_sized(sized: list[tuple[int, Path]], max_workers: int | None = None) -> tuple[int, int]:
    """Read pre-stat'ed files concurrently, largest-first; see prewarm_files."""
    sized = sorted(sized, key=lambda item: item[0], reverse=True)

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


def prewarm_files(paths, max_workers: int | None = None) -> tuple[int, int]:
    """Read ``paths`` concurrently, discarding the bytes.

    Each file is one unit of work (streamed sequentially within itself, so
    readahead still engages), dispatched largest-first across the reader
    pool. Duplicate paths are read once. Returns ``(n_files, n_bytes)``
    actually read. Files that vanish or can't be opened are skipped — the
    goal is a warm cache, not an audit.
    """
    return _read_sized(_stat_files(paths), max_workers)


def _fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


# cgroup v1's "no limit" placeholder is PAGE_COUNTER_MAX (~2**63); anything
# in that region is a sentinel, not a limit a job scheduler configured.
_CGROUP_NO_LIMIT = 1 << 60


def _cgroup_memory_limit(
    proc_cgroup: str = "/proc/self/cgroup", cgroup_root: str = "/sys/fs/cgroup"
) -> int | None:
    """This process's effective cgroup memory limit in bytes, or None.

    Page-cache pages are charged to the job's memory cgroup, so the limit —
    not node RAM — is the warmth budget. Checks every ancestor of our cgroup
    because schedulers typically set the limit on the job slice, not the
    leaf (SLURM's ``--mem``). Handles v2 (``memory.max``) and legacy v1
    (``memory/…/memory.limit_in_bytes``); best-effort, None on any surprise.
    """
    try:
        lines = Path(proc_cgroup).read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, cgroup_path = parts
        if controllers == "":  # v2 unified hierarchy
            base, filename = Path(cgroup_root), "memory.max"
        elif "memory" in controllers.split(","):
            base, filename = Path(cgroup_root) / "memory", "memory.limit_in_bytes"
        else:
            continue
        node = base / cgroup_path.lstrip("/")
        for current in (node, *node.parents):
            try:
                raw = (current / filename).read_text().strip()
            except OSError:
                raw = ""
            if raw.isdigit() and int(raw) < _CGROUP_NO_LIMIT:
                limits.append(int(raw))
            if current == base:
                break
    return min(limits) if limits else None


def prewarm_from_spec(spec: dict, log=None, label: str = "[Worker]") -> None:
    """Warm the page cache for a worker spawn spec; never raises.

    The one-line summary goes to ``log`` (default stderr, so it lands in
    the worker's captured output and any post-mortem) — its duration is a
    direct read on filesystem health: seconds when warm, and when cold it
    replaces an hours-long stall with a visible, bounded read. The weights
    portion is reported separately, tagged with the tier that resolved it
    (#178) — field data for retiring the heuristic once manifest records
    are universal. ``label`` prefixes each line: the default is the worker
    wrapper's; the ``rootstock stage`` prologue fallback passes its own.
    """
    if os.environ.get("ROOTSTOCK_NO_PREWARM"):
        return
    if log is None:
        log = sys.stderr
    try:
        began = time.monotonic()
        env_sized = _stat_files(_iter_env_files(spec))
        env_paths = {path for _, path in env_sized}
        weight_sized = [
            item for item in _stat_files(_iter_weight_files(spec)) if item[1] not in env_paths
        ]
        weight_bytes = sum(size for size, _ in weight_sized)

        # Cached pages are charged to the job's memory cgroup: a working set
        # bigger than the limit means early-warmed pages are evicted before
        # the worker reads them, silently re-paying the cold-fault tax
        # (observed on Delta, 2026-07-29). Warn up front — before the reads —
        # so the line survives even if the warm-up itself then stalls.
        expected = sum(size for size, _ in env_sized) + weight_bytes
        limit = _cgroup_memory_limit()
        if limit is not None and expected > limit:
            print(
                f"{label} Warning: expected cold working set "
                f"{_fmt_bytes(expected)} exceeds this job's memory limit "
                f"{_fmt_bytes(limit)}; warmed pages will be evicted before "
                f"the worker reads them — request more memory for the job",
                file=log,
                flush=True,
            )

        n_files, n_bytes = _read_sized(env_sized + weight_sized)
        elapsed = time.monotonic() - began

        summary = (
            f"{label} Prewarmed page cache: {n_files} files, "
            f"{_fmt_bytes(n_bytes)} in {elapsed:.1f}s"
        )
        tier = spec.get("prewarm_weights_tier")
        if tier == "none":
            summary += "; weights: none recorded"
        elif tier or weight_sized:
            # No tier annotation means spawn_in_env didn't fill the paths:
            # either the caller supplied its own prewarm_paths, or only a
            # :custom checkpoint_path contributed. Label accordingly — this
            # line is the field data for retiring the heuristic, so tags
            # must not lie about provenance.
            if not tier:
                tier = "custom" if not spec.get("prewarm_paths") else "caller"
            summary += f"; weights: {_fmt_bytes(weight_bytes)} ({tier})"
        print(summary, file=log, flush=True)
    except Exception as exc:  # noqa: BLE001 - never take the worker down
        try:
            print(f"{label} Prewarm skipped: {type(exc).__name__}: {exc}", file=log, flush=True)
        except Exception:
            pass
