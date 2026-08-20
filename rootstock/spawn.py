"""
Run code inside a pre-built environment.

One mechanism serves every "execute in an env" need: the worker spawned by
``RootstockServer``, the standalone worker started by ``rootstock serve``,
and the checkpoint download run by ``rootstock add``.

The wrapper sources below are static. Every runtime value — checkpoint,
device, socket path, setup kwargs, even the env directory — travels through
a JSON sidecar passed as ``argv[1]``, so no value is ever interpolated into
Python source and there is no repr()/escaping business for arbitrary values.

``spawn_in_env`` is a context manager: it stages the wrapper and sidecar in
a private temp directory and yields the command to run. The caller executes
it however it needs (``subprocess.run``, ``Popen``) but must keep the
context open for the life of the process — the worker reads both files at
startup, and the exit path removes the directory. Cleanup is exception-safe;
a SIGKILL still leaks the directory into TMPDIR, which is at least
node-local and auto-cleaned rather than shared.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .environment import get_checkpoint_prewarm_paths, get_env_python, get_model_cache_env

logger = logging.getLogger(__name__)

WORKER_WRAPPER = """\
import json, os, sys

with open(sys.argv[1]) as f:
    spec = json.load(f)

# Warm the page cache before the heavy imports below fault their way
# through multi-GB mmap'd libraries at network-round-trip speed (#167).
# The helpers are staged next to this wrapper; best-effort, never fatal.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
try:
    import prewarm
    prewarm.prewarm_from_spec(spec)
except ImportError:
    pass  # wrapper run without staged helper (e.g. tests); warming is optional
try:
    # Record which weight files setup() touches (#177); a no-op unless the
    # spec requests capture. The hook must be live before env_source's
    # imports so nothing loaded at import time slips past it.
    import weights_capture
    weights_capture.begin(spec)
except ImportError:
    weights_capture = None
finally:
    sys.path.remove(_here)

sys.path.insert(0, spec["env_dir"])
from rootstock.worker import run_worker

# User-supplied weights (:custom checkpoints) load through the env's opt-in
# setup_from_path hook; the path travels through run_worker's existing
# checkpoint parameter, so workers frozen inside built envs need no change.
# The imports are one-sided: canonical mode must not require the hook.
if spec.get("checkpoint_path"):
    from env_source import setup_from_path as setup_fn
    target = spec["checkpoint_path"]
else:
    from env_source import setup as setup_fn
    target = spec["checkpoint"]

# setup runs deep inside run_worker's socket loop, so the capture record is
# written by a wrapped setup_fn the moment the load completes.
if weights_capture is not None:
    setup_fn = weights_capture.wrap_setup(setup_fn, spec)

run_worker(
    setup_fn=setup_fn,
    checkpoint=target,
    device=spec["device"],
    socket_path=spec["socket_path"],
    setup_kwargs=spec["setup_kwargs"],
)
"""

DOWNLOAD_WRAPPER = """\
import json, os, sys

with open(sys.argv[1]) as f:
    spec = json.load(f)

# Same cold-cache prewarm as WORKER_WRAPPER — setup() below imports the
# same multi-GB libraries before it can download anything.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
try:
    import prewarm
    prewarm.prewarm_from_spec(spec)
except ImportError:
    pass  # wrapper run without staged helper (e.g. tests); warming is optional
try:
    # Same weight capture as WORKER_WRAPPER (#177) — a fresh download's
    # setup() writes and then loads the weights, so its touched set is the
    # same working set a verify-time load produces.
    import weights_capture
    weights_capture.begin(spec)
except ImportError:
    weights_capture = None
finally:
    sys.path.remove(_here)

sys.path.insert(0, spec["env_dir"])
from env_source import setup

# wrap_setup, not a trailing finalize(): the maps probe must scan while the
# calculator is still referenced. A discarded return value dies immediately
# under CPython refcounting, and mmap-backed weights (safetensors) are
# munmap'd with it — the wrapper's local reference holds them alive.
if weights_capture is not None:
    setup = weights_capture.wrap_setup(setup, spec)

setup(spec["checkpoint"], spec["device"], **spec["setup_kwargs"])
"""


@dataclass
class SpawnCommand:
    """How to execute the staged wrapper: argv, environment, and cwd."""

    cmd: list[str]
    env: dict[str, str]
    cwd: str


@contextmanager
def spawn_in_env(
    root: Path,
    env_name: str,
    wrapper_source: str,
    payload: dict,
    cache_root: Path | None = None,
    offline: bool = False,
) -> Iterator[SpawnCommand]:
    """
    Stage ``wrapper_source`` + a JSON sidecar for ``payload`` and yield the
    command that runs them with the env's Python.

    Args:
        root: Rootstock install root.
        env_name: Name of the pre-built environment.
        wrapper_source: One of the static wrapper sources in this module.
        payload: JSON-serializable values the wrapper reads from the sidecar.
            ``env_dir`` is filled in here, and when the payload names a
            ``checkpoint``, so are ``prewarm_paths``/``prewarm_weights_tier``
            (the checkpoint's weight files from the manifest record, or a
            heuristic cache scan — see get_checkpoint_prewarm_paths, #178);
            a caller-supplied ``prewarm_paths`` is left alone. Everything
            else is the caller's. A ``weights_capture`` entry of
            ``{"result_path", "cache_root"}`` makes the wrapper record which
            weight files setup() touches (see weights_capture.py); the
            caller reads ``result_path`` after the process exits.
        cache_root: Optional split cache root (see get_model_cache_env).
        offline: Forbid HuggingFace Hub network access (HF_HUB_OFFLINE=1).
            Workers serve weights pre-fetched by ``rootstock add`` and often
            run on compute nodes with no internet, where the hub's freshness
            HEAD request can crash setup() instead of falling back to the
            cached file. Downloads leave this off.

    Raises:
        RuntimeError: the environment is not built.
    """
    root = Path(root)
    env_python = get_env_python(root, env_name)
    env_dir = root / "envs" / env_name

    # Fill the weight-prewarm hint here rather than in each caller: every
    # spawn (calculator, verify, serve, add) benefits, and this is the same
    # choke point that already owns env_dir and the cache env vars.
    # Best-effort by contract — a failed lookup must never fail the spawn.
    # Download spawns get the record tier only: their weight record is
    # written after the download, so the heuristic would cold-read whole
    # family cache dirs (typically on a login node) for weights the
    # download may be about to (re)write.
    if payload.get("checkpoint") and "prewarm_paths" not in payload:
        try:
            paths, tier = get_checkpoint_prewarm_paths(
                root,
                env_name,
                payload["checkpoint"],
                cache_root,
                allow_heuristic=wrapper_source != DOWNLOAD_WRAPPER,
            )
        except Exception:
            logger.debug("prewarm path lookup failed; spawning without", exc_info=True)
        else:
            if tier is not None:
                payload = {**payload, "prewarm_paths": paths, "prewarm_weights_tier": tier}

    env = os.environ.copy()
    env.update(get_model_cache_env(root, cache_root))
    if offline:
        env["HF_HUB_OFFLINE"] = "1"

    # Prepend the env's own lib/ to LD_LIBRARY_PATH. Some GPU wheels ship native
    # runtime libraries there and load them at import via the dynamic linker --
    # e.g. PyTorch's Intel XPU build (Aurora) bundles its oneAPI/SYCL runtime in
    # <venv>/lib and `import torch` fails without it on the path. Harmless for
    # envs that don't need it (a venv lib/ dir with no such libraries).
    env_lib = env_dir / "lib"
    if env_lib.is_dir():
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{env_lib}{os.pathsep}{prev}" if prev else str(env_lib)

    tmp_dir = tempfile.mkdtemp(prefix="rootstock_spawn_")
    try:
        wrapper = Path(tmp_dir) / "wrapper.py"
        wrapper.write_text(wrapper_source)
        # These helpers run inside the env's (different) Python, so they
        # travel as staged source next to the wrapper rather than being
        # imported from the client's installation.
        for helper in ("prewarm.py", "weights_capture.py"):
            (Path(tmp_dir) / helper).write_text((Path(__file__).parent / helper).read_text())
        sidecar = Path(tmp_dir) / "spec.json"
        sidecar.write_text(json.dumps({**payload, "env_dir": str(env_dir)}))

        # cwd=env_dir keeps runtime relative-path resolution inside the env
        # rather than the caller's CWD, where a file named after the model
        # package (e.g. mace.py in environments/) would shadow the installed
        # package on import.
        yield SpawnCommand(
            cmd=[str(env_python), str(wrapper), str(sidecar)],
            env=env,
            cwd=str(env_dir),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
