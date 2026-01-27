"""
Environment management for Rootstock.

This module handles:
- Resolving environment names to file paths
- Generating wrapper scripts for `uv run`
- Providing spawn commands for worker processes
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .pep723 import parse_pep723_metadata

# Template for generated wrapper scripts
# This script is run via `uv run --with rootstock wrapper.py`
WRAPPER_TEMPLATE = """# /// script
# requires-python = "{requires_python}"
# dependencies = {dependencies_json}
# ///

import sys
sys.path.insert(0, "{env_dir}")

from {env_module} import setup
from rootstock.worker import run_worker

run_worker(
    setup_fn=setup,
    model="{model}",
    device="{device}",
    socket_path="{socket_path}",
)
"""


def get_rootstock_path() -> Path:
    """
    Get the path to the rootstock package.

    This is used for `uv run --with /path/to/rootstock` to inject
    rootstock into the worker's environment.
    """
    import rootstock

    return Path(rootstock.__file__).parent.parent


class EnvironmentManager:
    """
    Manages rootstock environment files and worker spawning.

    An environment file is a Python script with PEP 723 metadata
    and a `setup(model, device)` function that returns an ASE calculator.
    """

    def __init__(self, root: Path | str | None = None):
        """
        Initialize the environment manager.

        Args:
            root: Root directory for environments and cache.
                  If None, environments can only be loaded by direct path.
        """
        self.root = Path(root) if root else None
        self._temp_files: list[Path] = []

    def resolve_environment(self, environment: str) -> Path:
        """
        Resolve an environment name or path to an actual file path.

        Args:
            environment: Either an absolute path to an environment file,
                        or the name of a registered environment.

        Returns:
            Path to the environment file.

        Raises:
            FileNotFoundError: If the environment cannot be found.
        """
        # If it's an absolute path, use it directly
        if environment.startswith("/"):
            path = Path(environment)
            if not path.exists():
                raise FileNotFoundError(f"Environment file not found: {path}")
            return path

        # Otherwise, look in {root}/environments/
        if self.root is None:
            raise ValueError(
                f"Cannot resolve environment '{environment}' without a root directory. "
                "Either provide root= or use an absolute path."
            )

        # Try with and without .py extension
        env_dir = self.root / "environments"
        for name in [environment, f"{environment}.py"]:
            path = env_dir / name
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Environment '{environment}' not found in {env_dir}. "
            f"Available environments: {list(env_dir.glob('*.py')) if env_dir.exists() else []}"
        )

    def generate_wrapper(
        self,
        env_path: Path,
        model: str,
        device: str,
        socket_path: str,
    ) -> Path:
        """
        Generate a wrapper script for the given environment.

        The wrapper script includes the PEP 723 metadata from the environment
        file, imports the setup function, and calls run_worker().

        Args:
            env_path: Path to the environment file
            model: Model identifier to pass to setup()
            device: Device string to pass to setup()
            socket_path: Unix socket path for IPC

        Returns:
            Path to the generated wrapper script (temp file).
        """
        content = env_path.read_text()
        metadata = parse_pep723_metadata(content)

        if metadata is None:
            raise ValueError(f"No PEP 723 metadata in {env_path}")

        # Extract metadata
        requires_python = metadata.get("requires-python", ">=3.10")
        dependencies = metadata.get("dependencies", [])

        # Module name is the filename without extension
        env_module = env_path.stem

        # Format dependencies as JSON array for TOML
        dependencies_json = json.dumps(dependencies)

        # Generate wrapper content
        wrapper_content = WRAPPER_TEMPLATE.format(
            requires_python=requires_python,
            dependencies_json=dependencies_json,
            env_dir=str(env_path.parent),
            env_module=env_module,
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

    def get_spawn_command(self, wrapper_path: Path) -> list[str]:
        """
        Get the command to spawn a worker via uv run.

        Args:
            wrapper_path: Path to the generated wrapper script.

        Returns:
            Command list for subprocess.Popen, e.g.:
            ["uv", "run", "--with", "/path/to/rootstock", "/tmp/wrapper.py"]

        Note:
            We don't set --cache-dir here because some filesystems (like Modal volumes)
            don't support uv's lock files. The uv cache uses its default location,
            while HuggingFace cache is set via HF_HOME in get_environment_variables().
        """
        rootstock_path = get_rootstock_path()
        return ["uv", "run", "--with", str(rootstock_path), str(wrapper_path)]

    def get_environment_variables(self) -> dict[str, str]:
        """
        Get environment variables to set for the worker process.

        Returns:
            Dict of environment variables, including HF_HOME if root is set.

        Note:
            We only set HF_HOME (HuggingFace cache) on the root volume, not UV_CACHE_DIR.
            This is because some filesystems (like Modal volumes) don't support uv's
            lock files. The HuggingFace cache stores large model weights that benefit
            from persistence, while uv's pip cache can be rebuilt quickly.
        """
        import os

        env = os.environ.copy()

        if self.root is not None:
            # Set HuggingFace cache directory (for model weights)
            hf_home = self.root / "cache" / "huggingface"
            hf_home.mkdir(parents=True, exist_ok=True)
            env["HF_HOME"] = str(hf_home)

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
    List registered environments.

    Args:
        root: Root directory containing environments/

    Returns:
        List of (name, path) tuples for each environment.
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
