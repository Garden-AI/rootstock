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
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .environment import get_env_python, get_model_cache_env

WORKER_WRAPPER = """\
import json, sys

with open(sys.argv[1]) as f:
    spec = json.load(f)

sys.path.insert(0, spec["env_dir"])
from env_source import setup
from rootstock.worker import run_worker

run_worker(
    setup_fn=setup,
    checkpoint=spec["checkpoint"],
    device=spec["device"],
    socket_path=spec["socket_path"],
    setup_kwargs=spec["setup_kwargs"],
)
"""

DOWNLOAD_WRAPPER = """\
import json, sys

with open(sys.argv[1]) as f:
    spec = json.load(f)

sys.path.insert(0, spec["env_dir"])
from env_source import setup

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
) -> Iterator[SpawnCommand]:
    """
    Stage ``wrapper_source`` + a JSON sidecar for ``payload`` and yield the
    command that runs them with the env's Python.

    Args:
        root: Rootstock install root.
        env_name: Name of the pre-built environment.
        wrapper_source: One of the static wrapper sources in this module.
        payload: JSON-serializable values the wrapper reads from the sidecar.
            ``env_dir`` is filled in here; everything else is the caller's.
        cache_root: Optional split cache root (see get_model_cache_env).

    Raises:
        RuntimeError: the environment is not built.
    """
    root = Path(root)
    env_python = get_env_python(root, env_name)
    env_dir = root / "envs" / env_name

    env = os.environ.copy()
    env.update(get_model_cache_env(root, cache_root))

    tmp_dir = tempfile.mkdtemp(prefix="rootstock_spawn_")
    try:
        wrapper = Path(tmp_dir) / "wrapper.py"
        wrapper.write_text(wrapper_source)
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
