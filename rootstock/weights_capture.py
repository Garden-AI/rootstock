"""Record which model-weight files a checkpoint load actually touches.

Weight locations vary per library and guessing them is a losing game (#177):
well-behaved libraries land under ``{cache_root}/cache`` via XDG_CACHE_HOME /
HF_HOME, but e.g. fairchem hardcodes ``~/.cache`` so UMA's 14 GB lives under
the redirected ``{cache_root}/home``. Rootstock controls every sanctioned
path by which weights arrive and get loaded — ``rootstock add`` runs
``setup()`` in DOWNLOAD_WRAPPER, verify/smoke-test load the same weights in
WORKER_WRAPPER — so instead of discovering cache layouts we observe what
those runs touch and write it down for the manifest.

Two probes, both stdlib:

1. A ``sys.addaudithook`` on ``open`` events — every Python-side open under
   ``{cache_root}/cache`` and ``{cache_root}/home`` while capture is active.
2. After ``setup()`` returns, a scan of ``/proc/self/maps`` for file
   mappings under the same trees — mmap'd reads bypass Python ``open``
   entirely (safetensors' native reader), and mmap'd weights are exactly
   the slow-fault case prewarm exists for. Linux-only; silently skipped
   elsewhere.

Like ``prewarm.py``, this module is staged next to the wrapper by
``spawn_in_env`` and runs inside the *target env's* Python — it must stay
stdlib-only and compatible with any Python an env might ship. Capture is
requested via ``spec["weights_capture"] = {"result_path", "cache_root"}``;
without that key every entry point is a no-op. Best-effort by design: a
capture failure logs and never takes down the download or the worker.
"""

from __future__ import annotations

import json
import os
import sys

# Capture state for the current process. sys.addaudithook is irreversible,
# so the hook closes over this dict and goes dormant via "active" instead of
# being removed; a long-lived worker stops paying for capture (beyond one
# flag check per open) once finalize() has run.
_state: dict | None = None


def _scope_prefixes(cache_root: str) -> tuple[str, ...]:
    """Directory prefixes under which touched files count as weights.

    Both the spec-provided form and its realpath are included: the audit
    hook sees whatever path the library opened (built from env vars that use
    the spec form), while /proc/self/maps records canonical paths.
    """
    prefixes = []
    for base in dict.fromkeys((cache_root, os.path.realpath(cache_root))):
        for sub in ("cache", "home"):
            prefixes.append(os.path.join(base, sub) + os.sep)
    return tuple(prefixes)


def begin(spec: dict) -> None:
    """Install the open-event audit hook when the spec requests capture.

    Called before the env's heavy imports so nothing that loads during
    ``import env_source`` can slip past the hook.
    """
    global _state
    cfg = spec.get("weights_capture")
    if not cfg:
        return
    try:
        state = {
            "scopes": _scope_prefixes(cfg["cache_root"]),
            "cache_root": cfg["cache_root"],
            "result_path": cfg["result_path"],
            "seen": set(),
            "active": True,
        }
        scopes = state["scopes"]
        seen = state["seen"]

        def _hook(event: str, args: tuple) -> None:
            # An exception raised here propagates into whatever triggered
            # the event — swallow everything.
            try:
                if event == "open" and state["active"]:
                    path = args[0]
                    if isinstance(path, str) and path.startswith(scopes):
                        seen.add(path)
            except Exception:
                pass

        sys.addaudithook(_hook)
        _state = state
    except Exception as exc:  # noqa: BLE001 - capture must never be fatal
        _log(f"Weight capture disabled: {type(exc).__name__}: {exc}")
        _state = None


def parse_maps(maps_text: str, scopes: tuple[str, ...]) -> set[str]:
    """Extract in-scope file paths from /proc/self/maps content.

    Each line is ``addr perms offset dev inode [pathname]``; anonymous
    mappings have no pathname and unlinked files carry a ``(deleted)``
    suffix (nothing to prewarm there — skipped).
    """
    paths: set[str] = set()
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        path = parts[5].strip()
        if path.startswith(scopes) and not path.endswith(" (deleted)"):
            paths.add(path)
    return paths


def _mapped_files(scopes: tuple[str, ...]) -> set[str]:
    try:
        with open("/proc/self/maps") as f:
            return parse_maps(f.read(), scopes)
    except OSError:
        return set()  # not Linux (or /proc unavailable) — audit hook only


def finalize(spec: dict) -> None:
    """Merge both probes, stat sizes, and write the capture result JSON.

    Runs right after ``setup()`` returns — the process may live on for a
    whole verification run (or die abruptly at server.stop()), so the
    result must be on disk the moment the load working set is known.
    """
    global _state
    state = _state
    if state is None or not spec.get("weights_capture"):
        return
    try:
        touched = set(state["seen"]) | _mapped_files(state["scopes"])
        state["active"] = False

        real_root = os.path.realpath(state["cache_root"])
        files: dict[str, int] = {}
        for path in touched:
            # Canonicalize so a file seen through a symlink (HF snapshots/
            # point into blobs/) and through its real path records once, as
            # the physical file prewarm would read.
            real = os.path.realpath(path)
            rel = os.path.relpath(real, real_root)
            if rel.startswith(".."):
                continue  # symlink escaped the cache tree
            try:
                if not os.path.isfile(real):
                    continue
                size = os.stat(real).st_size
            except OSError:
                continue  # vanished (e.g. a download's temp file)
            if size == 0:
                continue  # lock files and friends — nothing to prewarm
            files[rel] = size

        result = {"files": [{"path": p, "size": s} for p, s in sorted(files.items())]}
        with open(state["result_path"], "w") as f:
            json.dump(result, f)

        total = sum(files.values())
        _log(f"Weight capture: {len(files)} files, {total / 1e6:.0f} MB")
    except Exception as exc:  # noqa: BLE001 - capture must never be fatal
        _log(f"Weight capture failed: {type(exc).__name__}: {exc}")
    finally:
        state["active"] = False
        _state = None


def wrap_setup(setup_fn, spec: dict):
    """Wrap a setup function so finalize() runs the moment it returns.

    Used by WORKER_WRAPPER, where setup happens deep inside run_worker's
    socket loop and the wrapper never regains control; the record is
    written before the worker even connects. When the spec doesn't request
    capture, returns ``setup_fn`` unchanged. A setup that raises writes no
    record — a failed load has no working set worth recording.
    """
    if not spec.get("weights_capture"):
        return setup_fn

    def _capturing_setup(*args, **kwargs):
        calculator = setup_fn(*args, **kwargs)
        finalize(spec)
        return calculator

    return _capturing_setup


def _log(message: str) -> None:
    try:
        print(f"[Worker] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass
