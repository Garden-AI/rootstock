"""
Environment management for Rootstock.

This module handles:
- Locating pre-built virtual environments and their interpreters
- Cache-redirection environment variables for worker processes
- Checkpoint discovery from installed env sources

Running code inside an env lives in rootstock.spawn.
"""

from __future__ import annotations

import ast
import os
import shutil
from pathlib import Path

from .exceptions import RootstockError


class CheckpointNotFoundError(RootstockError, LookupError):
    """Raised when a canonical checkpoint id is not declared by any installed env."""


def get_user_cache_dir() -> Path:
    """
    Per-user directory for runtime write-back caches.

    Shared installs are world-*readable* but only maintainer-writable, so
    anything a worker writes at runtime (compiled Triton/Inductor kernels,
    torch C++ extension builds, NVIDIA compute cache, bytecode) must land
    somewhere the calling user owns — not in the shared cache root.

    Resolved from the real caller's home (before any HOME redirection is
    applied); override with ROOTSTOCK_USER_CACHE_DIR.
    """
    override = os.environ.get("ROOTSTOCK_USER_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "rootstock"


def get_model_cache_env(root: Path, cache_root: Path | None = None) -> dict[str, str]:
    """
    Get environment variables to redirect model downloads to a shared cache
    and runtime write-back to a per-user cache.

    HOME is redirected for libraries that hardcode `~/` for caching (e.g.,
    FAIRChem). XDG_CACHE_HOME and HF_HOME catch the well-behaved libraries.
    Those point at the *shared* cache, which non-maintainers can only read —
    so every cache a library writes at runtime (Triton/Inductor kernels,
    torch extension builds, NVIDIA compute cache, bytecode) is redirected to
    a per-user directory instead. Per-user vars respect values already set
    in the caller's environment (e.g. TRITON_CACHE_DIR on node-local SSD).

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

    user_cache = get_user_cache_dir()
    per_user_defaults = {
        "TRITON_CACHE_DIR": user_cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": user_cache / "torchinductor",
        "TORCH_EXTENSIONS_DIR": user_cache / "torch_extensions",
        "CUDA_CACHE_PATH": user_cache / "nv" / "ComputeCache",
        # NVIDIA Warp (a dep of nvalchemi-toolkit-ops, used by the tensornet
        # env) compiles kernels into $XDG_CACHE_HOME/warp — i.e. the shared
        # cache — unless WARP_CACHE_PATH points elsewhere.
        "WARP_CACHE_PATH": user_cache / "warp",
        "PYTHONPYCACHEPREFIX": user_cache / "pycache",
        "XDG_CONFIG_HOME": user_cache / "config",
        "MPLCONFIGDIR": user_cache / "matplotlib",
        # cached_path (an orb-models dep) ignores XDG_CACHE_HOME and defaults
        # to ~/.cache/cached_path under the redirected HOME — i.e. the shared
        # root — where it takes a FileLock even on warm cache hits (#67).
        # Point it per-user; envs that serve from the shared cache must hand
        # cached_path a *local* file (see nvidia_configs/orb.py), which it
        # returns without locking.
        "CACHED_PATH_CACHE_ROOT": user_cache / "cached_path",
    }

    env = {
        "HOME": str(home_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
    }
    for var, default in per_user_defaults.items():
        env[var] = os.environ.get(var) or str(default)

    # Preserve HuggingFace authentication tokens from the caller's environment.
    # These allow access to gated models.
    for auth_var in ("HF_TOKEN", "HF_USER_ACCESS_TOKEN"):
        if auth_var in os.environ:
            env[auth_var] = os.environ[auth_var]

    return env


def get_env_python(root: Path | str, env_name: str) -> Path:
    """
    Get path to Python executable for a pre-built environment.

    Args:
        root: Install root directory (envs, environments, manifest).
        env_name: Name of the environment (e.g., "mace_env").

    Returns:
        Path to the environment's Python executable.

    Raises:
        RuntimeError: If the environment is not built.
    """
    root = Path(root)
    env_python = root / "envs" / env_name / "bin" / "python"

    if not env_python.exists():
        envs_dir = root / "envs"
        if envs_dir.exists():
            available = [p.name for p in envs_dir.iterdir() if p.is_dir()]
        else:
            available = []

        raise RuntimeError(
            f"Environment '{env_name}' not built. "
            f"Run: rootstock install {env_name} --root {root}\n"
            f"Available environments: {available}"
        )

    return env_python


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
            raise ValueError(f"{env_source_path}: CHECKPOINTS must be a dict literal.")
        result: dict[str, str] = {}
        for k_node, v_node in zip(value.keys, value.values):
            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                raise ValueError(f"{env_source_path}: CHECKPOINTS keys must be string literals.")
            if not (isinstance(v_node, ast.Constant) and isinstance(v_node.value, str)):
                raise ValueError(f"{env_source_path}: CHECKPOINTS values must be string literals.")
            result[k_node.value] = v_node.value
        return result
    raise ValueError(
        f"{env_source_path}: missing module-level CHECKPOINTS dict. "
        f"Each env file must declare a `CHECKPOINTS: dict[str, str]` mapping "
        f"canonical checkpoint ids to upstream library strings."
    )


def declares_setup_from_path(env_source_path: Path) -> bool:
    """
    Return True when the env source declares a module-level ``setup_from_path``.

    ``setup_from_path(path, device="cuda", **kwargs)`` is the opt-in hook that
    lets an env load user-supplied weights files (local checkpoints). Presence
    is all that's checked — the signature is trusted the same way ``setup``'s
    is.
    """
    tree = ast.parse(env_source_path.read_text(), filename=str(env_source_path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "setup_from_path"
        for node in tree.body
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


def find_env_for_checkpoint(root: Path | str, checkpoint_id: str) -> tuple[str, dict[str, str]]:
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
            f"  {env}: {', '.join(ids) if ids else '(none)'}" for env, ids in declared.items()
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
