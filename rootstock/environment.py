"""
Environment management for Rootstock.

This module handles:
- Managing pre-built virtual environments
- Generating wrapper scripts for worker processes
- Providing spawn commands for worker processes
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def get_model_cache_env(root: Path) -> dict[str, str]:
    """
    Get environment variables to redirect model downloads to shared cache.

    We set XDG_CACHE_HOME as a catch-all for libraries following the XDG spec
    (MACE uses ~/.cache/mace/), plus explicit HF vars for HuggingFace.

    Args:
        root: Rootstock root directory.

    Returns:
        Dict of environment variables for model caching.
    """
    cache_dir = root / "cache"
    return {
        # XDG base directory - catches MACE and other well-behaved libraries
        "XDG_CACHE_HOME": str(cache_dir),
        # HuggingFace explicit (some tools check these before XDG)
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
    }


# Simplified wrapper template for pre-built environments
# No PEP 723 metadata needed since dependencies are already installed
WRAPPER_TEMPLATE = """
import sys
sys.path.insert(0, "{env_dir}")
from env_source import setup
from rootstock.worker import run_worker

run_worker(
    setup_fn=setup,
    model="{model}",
    device="{device}",
    socket_path="{socket_path}",
)
"""


class EnvironmentManager:
    """
    Manages pre-built rootstock environments and worker spawning.

    In v0.4+, environments are pre-built virtual environments located in
    {root}/envs/{env_name}/. The environment source file is copied into
    the venv as env_source.py during build.
    """

    def __init__(self, root: Path | str):
        """
        Initialize the environment manager.

        Args:
            root: Root directory for environments and cache.
        """
        self.root = Path(root)
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
        model: str,
        device: str,
        socket_path: str,
    ) -> Path:
        """
        Generate a wrapper script for the given environment.

        Args:
            env_name: Name of the pre-built environment
            model: Model identifier to pass to setup()
            device: Device string to pass to setup()
            socket_path: Unix socket path for IPC

        Returns:
            Path to the generated wrapper script (temp file).
        """
        env_dir = self.root / "envs" / env_name

        # Generate wrapper content
        wrapper_content = WRAPPER_TEMPLATE.format(
            env_dir=str(env_dir),
            model=model,
            device=device,
            socket_path=socket_path,
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
        env.update(get_model_cache_env(self.root))
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
