"""
Environment management for Rootstock.

This module handles:
- Managing pre-built virtual environments
- Generating wrapper scripts for worker processes
- Providing spawn commands for worker processes
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
from pathlib import Path


class CheckpointNotFoundError(LookupError):
    """Raised when a canonical checkpoint id is not declared by any installed env."""


def get_model_cache_env(root: Path, cache_root: Path | None = None) -> dict[str, str]:
    """
    Get environment variables to redirect model downloads to a shared cache.

    HOME is redirected for libraries that hardcode `~/` for caching (e.g.,
    FAIRChem). XDG_CACHE_HOME and HF_HOME catch the well-behaved libraries.

    On most clusters the install root and the cache root coincide. On clusters
    where they live on different filesystems (e.g., Perlmutter — code on CFS,
    cache on PSCRATCH), pass a separate `cache_root`.

    Args:
        root: Rootstock install root.
        cache_root: Optional separate root for the model-weight cache and
                    redirected HOME. Defaults to ``root``.

    Returns:
        Dict of environment variables for model caching.
    """
    base = cache_root if cache_root is not None else root
    cache_dir = base / "cache"
    home_dir = base / "home"
    return {
        "HOME": str(home_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
    }


# Simplified wrapper template for pre-built environments.
# No PEP 723 metadata needed since dependencies are already installed.
# setup_kwargs travel via a JSON sidecar file rather than being baked into
# the source — keeps us out of the repr()/escaping business for arbitrary values.
WRAPPER_TEMPLATE = """
import sys, json
sys.path.insert(0, "{env_dir}")
from env_source import setup
from rootstock.worker import run_worker

with open("{kwargs_path}") as f:
    setup_kwargs = json.load(f)

run_worker(
    setup_fn=setup,
    checkpoint="{checkpoint}",
    device="{device}",
    socket_path="{socket_path}",
    setup_kwargs=setup_kwargs,
)
"""


class EnvironmentManager:
    """
    Manages pre-built rootstock environments and worker spawning.

    In v0.4+, environments are pre-built virtual environments located in
    {root}/envs/{env_name}/. The environment source file is copied into
    the venv as env_source.py during build.
    """

    def __init__(self, root: Path | str, cache_root: Path | str | None = None):
        """
        Initialize the environment manager.

        Args:
            root: Install root directory (envs, environments, manifest).
            cache_root: Optional separate root for the model-weight cache and
                        redirected HOME. Defaults to ``root``.
        """
        self.root = Path(root)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._temp_files: list[Path] = []

    def get_env_python(self, env_name: str) -> Path:
        """
        Get path to Python executable for a pre-built environment.

        Args:
            env_name: Name of the environment (e.g., "mace_env").

        Returns:
            Path to the environment's Python executable.

        Raises:
            RuntimeError: If the environment is not built.
        """
        env_python = self.root / "envs" / env_name / "bin" / "python"

        if not env_python.exists():
            envs_dir = self.root / "envs"
            if envs_dir.exists():
                available = [p.name for p in envs_dir.iterdir() if p.is_dir()]
            else:
                available = []

            raise RuntimeError(
                f"Environment '{env_name}' not built. "
                f"Run: rootstock build {env_name} --root {self.root}\n"
                f"Available environments: {available}"
            )

        return env_python

    def generate_wrapper(
        self,
        env_name: str,
        checkpoint: str,
        device: str,
        socket_path: str,
        setup_kwargs: dict | None = None,
    ) -> Path:
        """
        Generate a wrapper script for the given environment.

        Args:
            env_name: Name of the pre-built environment
            checkpoint: Canonical checkpoint id to pass to setup()
            device: Device string to pass to setup()
            socket_path: Unix socket path for IPC
            setup_kwargs: Extra keyword arguments forwarded to setup() via JSON sidecar

        Returns:
            Path to the generated wrapper script (temp file).
        """
        env_dir = self.root / "envs" / env_name

        # Write setup_kwargs to a JSON sidecar that the wrapper reads at startup.
        kwargs_fd, kwargs_path = tempfile.mkstemp(suffix=".json", prefix="rootstock_kwargs_")
        with open(kwargs_fd, "w") as f:
            json.dump(setup_kwargs or {}, f)
        kwargs_path = Path(kwargs_path)
        self._temp_files.append(kwargs_path)

        # Generate wrapper content
        wrapper_content = WRAPPER_TEMPLATE.format(
            env_dir=str(env_dir),
            checkpoint=checkpoint,
            device=device,
            socket_path=socket_path,
            kwargs_path=str(kwargs_path),
        )

        # Write to temp file
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="rootstock_wrapper_")
        with open(fd, "w") as f:
            f.write(wrapper_content)

        tmp_path = Path(tmp_path)
        self._temp_files.append(tmp_path)

        return tmp_path

    def get_spawn_command(self, env_name: str, wrapper_path: Path) -> list[str]:
        """
        Get the command to spawn a worker using pre-built environment.

        Args:
            env_name: Name of the pre-built environment.
            wrapper_path: Path to the generated wrapper script.

        Returns:
            Command list for subprocess.Popen, e.g.:
            ["/vol/rootstock/envs/mace_env/bin/python", "/tmp/wrapper.py"]
        """
        env_python = self.get_env_python(env_name)
        return [str(env_python), str(wrapper_path)]

    def get_environment_variables(self) -> dict[str, str]:
        """
        Get environment variables to set for the worker process.

        Returns:
            Dict of environment variables for model caching.
        """
        env = os.environ.copy()
        env.update(get_model_cache_env(self.root, self.cache_root))
        return env

    def cleanup(self):
        """Clean up temporary wrapper files."""
        for path in self._temp_files:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_files.clear()

    def __del__(self):
        self.cleanup()


def check_uv_available() -> bool:
    """Check if uv is available in PATH."""
    return shutil.which("uv") is not None


def list_environments(root: Path | str) -> list[tuple[str, Path]]:
    """
    List registered environment source files.

    Args:
        root: Root directory containing environments/

    Returns:
        List of (name, path) tuples for each environment source file.
    """
    root = Path(root)
    env_dir = root / "environments"

    if not env_dir.exists():
        return []

    result = []
    for path in sorted(env_dir.glob("*.py")):
        name = path.stem
        result.append((name, path))

    return result


def parse_checkpoints_dict(env_source_path: Path) -> dict[str, str]:
    """
    Extract the module-level ``CHECKPOINTS: dict[str, str]`` literal from an env file.

    The dict maps canonical checkpoint ids → upstream library strings. Both keys
    and values must be string literals; anything else is an authoring error and
    raises ValueError.
    """
    tree = ast.parse(env_source_path.read_text(), filename=str(env_source_path))
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if not (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and targets[0].id == "CHECKPOINTS"
        ):
            continue
        if not isinstance(value, ast.Dict):
            raise ValueError(
                f"{env_source_path}: CHECKPOINTS must be a dict literal."
            )
        result: dict[str, str] = {}
        for k_node, v_node in zip(value.keys, value.values):
            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                raise ValueError(
                    f"{env_source_path}: CHECKPOINTS keys must be string literals."
                )
            if not (isinstance(v_node, ast.Constant) and isinstance(v_node.value, str)):
                raise ValueError(
                    f"{env_source_path}: CHECKPOINTS values must be string literals."
                )
            result[k_node.value] = v_node.value
        return result
    raise ValueError(
        f"{env_source_path}: missing module-level CHECKPOINTS dict. "
        f"Each env file must declare a `CHECKPOINTS: dict[str, str]` mapping "
        f"canonical checkpoint ids to upstream library strings."
    )


def list_declared_checkpoints(root: Path | str) -> dict[str, dict[str, str]]:
    """
    Walk ``{root}/envs/*/env_source.py`` and return ``{env_name: CHECKPOINTS}``
    for every installed env that declares a valid ``CHECKPOINTS`` dict.

    This is the single source of truth for "which canonical checkpoint ids can
    ``rootstock add`` accept". Envs whose source file is missing or whose
    ``CHECKPOINTS`` dict is malformed are silently skipped. Results are ordered
    by env name.
    """
    root = Path(root)
    envs_dir = root / "envs"
    declared: dict[str, dict[str, str]] = {}  # env_name -> {id: upstream}
    if not envs_dir.exists():
        return declared
    for env_dir in sorted(envs_dir.iterdir()):
        source = env_dir / "env_source.py"
        if not source.exists():
            continue
        try:
            declared[env_dir.name] = parse_checkpoints_dict(source)
        except ValueError:
            continue
    return declared


def find_env_for_checkpoint(
    root: Path | str, checkpoint_id: str
) -> tuple[str, dict[str, str]]:
    """
    Return ``(env_name, CHECKPOINTS)`` for the installed env that declares
    ``checkpoint_id``.

    Raises ``CheckpointNotFoundError`` with a message listing every canonical
    id declared by any installed env, plus a hint to ``rootstock install`` if
    nothing matches.
    """
    root = Path(root)
    declared = list_declared_checkpoints(root)
    for env_name, ckpts in declared.items():
        if checkpoint_id in ckpts:
            return env_name, ckpts

    if declared:
        listing = "\n".join(
            f"  {env}: {', '.join(ids) if ids else '(none)'}"
            for env, ids in declared.items()
        )
        msg = (
            f"No installed env declares checkpoint '{checkpoint_id}'.\n"
            f"Declared canonical ids by env:\n{listing}\n"
            f"If '{checkpoint_id}' belongs to an env you haven't installed yet, "
            f"run `rootstock install <env-file> --root {root}`."
        )
    else:
        msg = (
            f"No envs are installed at {root}. "
            f"Run `rootstock install <env-file> --root {root}` first."
        )
    raise CheckpointNotFoundError(msg)


def list_built_environments(root: Path | str) -> list[tuple[str, Path]]:
    """
    List pre-built environments.

    Args:
        root: Root directory containing envs/

    Returns:
        List of (name, path) tuples for each built environment.
    """
    root = Path(root)
    envs_dir = root / "envs"

    if not envs_dir.exists():
        return []

    result = []
    for path in sorted(envs_dir.iterdir()):
        if path.is_dir() and (path / "bin" / "python").exists():
            result.append((path.name, path))

    return result
