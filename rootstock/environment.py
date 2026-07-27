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
from dataclasses import dataclass
from pathlib import Path

from .exceptions import RootstockError


class CheckpointNotFoundError(RootstockError, LookupError):
    """Raised when a canonical checkpoint id is not declared by any installed env."""


class CustomWeightsError(RootstockError, ValueError):
    """A ``:custom`` checkpoint / ``weights`` pairing is invalid. Messages are
    user-presentable diagnoses that name the fix."""


# A CHECKPOINTS key ending in ":custom" (value None) declares a custom
# checkpoint for one model family: the id resolves to its hosting env like
# any other, but there are no shipped weights — the user supplies theirs via
# weights=, loaded through the env's setup_from_path hook. An env may declare
# several (one per user-facing family). ":" is otherwise reserved in
# checkpoint ids.
CUSTOM_CHECKPOINT_SUFFIX = ":custom"


def is_custom_checkpoint(checkpoint_id: str) -> bool:
    """True when ``checkpoint_id`` names a ``<family>:custom`` entry."""
    return checkpoint_id.endswith(CUSTOM_CHECKPOINT_SUFFIX)


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


def _parse_checkpoints(env_source_path: Path) -> tuple[dict[str, str], list[str]]:
    """AST-extract the module-level ``CHECKPOINTS`` literal, split into
    ``(canonical id -> upstream string, list of '<family>:custom' ids)``."""
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
        custom_ids: list[str] = []
        for k_node, v_node in zip(value.keys, value.values):
            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                raise ValueError(f"{env_source_path}: CHECKPOINTS keys must be string literals.")
            key = k_node.value
            if ":" in key:
                family = key.removesuffix(CUSTOM_CHECKPOINT_SUFFIX)
                if not is_custom_checkpoint(key) or not family or ":" in family:
                    raise ValueError(
                        f"{env_source_path}: CHECKPOINTS key '{key}': ':' is "
                        f"reserved in checkpoint ids — the only allowed form "
                        f"is '<family>{CUSTOM_CHECKPOINT_SUFFIX}', declaring "
                        f"that users may run their own weights for that "
                        f"model family."
                    )
                if not (isinstance(v_node, ast.Constant) and v_node.value is None):
                    raise ValueError(
                        f"{env_source_path}: CHECKPOINTS key '{key}' must map "
                        f"to None — a custom entry never names shipped "
                        f"weights; the user supplies them (weights= in "
                        f"Python, --weights on the CLI)."
                    )
                custom_ids.append(key)
                continue
            if not (isinstance(v_node, ast.Constant) and isinstance(v_node.value, str)):
                raise ValueError(f"{env_source_path}: CHECKPOINTS values must be string literals.")
            result[key] = v_node.value
        return result, custom_ids
    raise ValueError(
        f"{env_source_path}: missing module-level CHECKPOINTS dict. "
        f"Each env file must declare a `CHECKPOINTS: dict[str, str]` mapping "
        f"canonical checkpoint ids to upstream library strings."
    )


def parse_checkpoints_dict(env_source_path: Path) -> dict[str, str]:
    """
    Extract the module-level ``CHECKPOINTS: dict[str, str]`` literal from an
    env file — canonical entries only.

    The dict maps canonical checkpoint ids → upstream library strings; both
    sides must be string literals. ``<family>:custom`` entries (value
    ``None``) are validated and stripped: they declare a capability, not a
    downloadable checkpoint, and the consumers of this dict (``add``,
    smoke-test, status) must never see them. Anything else is an authoring
    error and raises ValueError.
    """
    return _parse_checkpoints(env_source_path)[0]


def parse_custom_checkpoint_ids(env_source_path: Path) -> list[str]:
    """
    The ``<family>:custom`` ids declared by an env source, in declaration
    order. Empty when the env doesn't declare support for user-supplied
    weights.
    """
    return _parse_checkpoints(env_source_path)[1]


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
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "setup_from_path"
        for node in tree.body
    )


def _suggest_custom_entry(root: Path | str, env_name: str, checkpoint: str) -> str:
    """Error text for ``weights`` arriving with a non-custom id: silently
    ignoring the file would run the shipped weights — plausible results,
    wrong model. Names the hosting env's ``:custom`` entry when it has one."""
    msg = (
        f"a weights file was supplied, but checkpoint '{checkpoint}' names "
        f"shipped weights, so it would be ignored."
    )
    env_source = Path(root) / "envs" / env_name / "env_source.py"
    custom_ids: list[str] = []
    if env_source.exists():
        try:
            custom_ids = parse_custom_checkpoint_ids(env_source)
        except ValueError:
            pass
    if len(custom_ids) == 1:
        return f"{msg} To run your own fine-tune, use checkpoint='{custom_ids[0]}'."
    if custom_ids:
        return (
            f"{msg} To run your own fine-tune, use the entry for its model "
            f"family: {', '.join(repr(c) for c in custom_ids)}."
        )
    return (
        f"{msg} Env '{env_name}' declares no '<family>{CUSTOM_CHECKPOINT_SUFFIX}' "
        f"CHECKPOINTS entry, so it does not support user-supplied weights — "
        f"ask the install maintainer."
    )


def bind_custom_weights(
    root: Path | str,
    env_name: str,
    checkpoint: str,
    weights: str | Path | None,
    setup_kwargs: dict | None = None,
) -> str | None:
    """
    Enforce the ``<family>:custom`` / ``weights`` pairing for a *resolved*
    checkpoint and, for a custom id, return the validated weights path (as
    ``str``) to hand to the worker as ``checkpoint_path``. Returns None for
    a canonical id without weights — nothing to bind.

    The pairing is structural typo-proofing: ASE's ``Calculator`` silently
    absorbs unknown kwargs, so a misspelled weights kwarg leaves ``weights``
    unset — with a ``:custom`` id there are no shipped weights to silently
    fall back to, so the typo errors here instead of running the wrong model.

    Custom-id checks, in order: weights supplied; no ``path`` in setup_kwargs
    (it's ``setup_from_path``'s first parameter — fail here, not as a
    TypeError inside the worker); the weights file exists; the *built* env
    source declares the ``setup_from_path`` hook (otherwise the failure would
    surface as an opaque ImportError inside the worker).
    """
    if not is_custom_checkpoint(checkpoint):
        if weights is None:
            return None
        raise CustomWeightsError(_suggest_custom_entry(root, env_name, checkpoint))
    if weights is None:
        raise CustomWeightsError(
            f"checkpoint '{checkpoint}' runs your own weights file — also "
            f"supply it (weights= in Python, --weights on the CLI)."
        )
    if setup_kwargs and "path" in setup_kwargs:
        raise CustomWeightsError(
            f"setup_kwargs cannot contain 'path' for a "
            f"'{CUSTOM_CHECKPOINT_SUFFIX}' checkpoint; the weights path is "
            f"passed at the top level."
        )
    # Resolve against the caller's CWD now: the worker is spawned with
    # cwd=env_dir, so a relative path passed through verbatim would be
    # re-resolved there — failing, or worse, loading a same-named file
    # inside the env dir.
    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise CustomWeightsError(
            f"weights file not found (or not a regular file): {weights_path}. "
            f"The path must be visible from the compute nodes."
        )
    env_source = Path(root) / "envs" / env_name / "env_source.py"
    if not (env_source.exists() and declares_setup_from_path(env_source)):
        raise CustomWeightsError(
            f"env '{env_name}' (hosting '{checkpoint}') does not declare "
            f"setup_from_path(path, device, **kwargs), which is required to "
            f"load user-supplied weights. Ask the install maintainer to "
            f"refresh the env sources — see docs/environments.md."
        )
    return str(weights_path)


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


def list_custom_checkpoints(root: Path | str) -> dict[str, list[str]]:
    """
    Walk ``{root}/envs/*/env_source.py`` and return
    ``{env_name: [<family>:custom ids]}`` for every installed env that
    declares at least one. Envs whose source is missing or malformed are
    silently skipped, mirroring ``list_declared_checkpoints``.
    """
    root = Path(root)
    envs_dir = root / "envs"
    declared: dict[str, list[str]] = {}
    if not envs_dir.exists():
        return declared
    for env_dir in sorted(envs_dir.iterdir()):
        source = env_dir / "env_source.py"
        if not source.exists():
            continue
        try:
            custom_ids = parse_custom_checkpoint_ids(source)
        except ValueError:
            continue
        if custom_ids:
            declared[env_dir.name] = custom_ids
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


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Where a checkpoint id points: its hosting env.

    A ``<family>:custom`` id resolves to its hosting env only — the user's
    weights path is bound at the call site (from ``weights=`` / ``--weights``),
    never during resolution; ``is_custom`` is the marker."""

    checkpoint: str
    env_name: str
    is_custom: bool = False


def _resolve_custom(root: Path | str, checkpoint_id: str) -> ResolvedCheckpoint:
    """
    Resolve a ``<family>:custom`` checkpoint via the entries installed envs
    declare in ``CHECKPOINTS``.

    The entry carries no weights (its value is ``None``) — the user's
    ``weights`` file is bound at the call site. Which env hosts the id is
    determined by which env's dict declares it, exactly like a canonical id.
    """
    declared = list_custom_checkpoints(root)
    for env_name, custom_ids in declared.items():
        if checkpoint_id in custom_ids:
            return ResolvedCheckpoint(checkpoint=checkpoint_id, env_name=env_name, is_custom=True)

    if declared:
        listing = "\n".join(f"  {env}: {', '.join(ids)}" for env, ids in declared.items())
        msg = (
            f"No installed env declares the custom checkpoint "
            f"'{checkpoint_id}'.\nDeclared '{CUSTOM_CHECKPOINT_SUFFIX}' "
            f"entries by env:\n{listing}"
        )
    else:
        msg = (
            f"No installed env declares a '<family>{CUSTOM_CHECKPOINT_SUFFIX}' "
            f"CHECKPOINTS entry, so user-supplied weights are not available "
            f"at {root}. Ask the install maintainer to refresh the env "
            f"sources — see docs/environments.md."
        )
    raise CheckpointNotFoundError(msg)


def resolve_checkpoint(root: Path | str, checkpoint_id: str) -> ResolvedCheckpoint:
    """
    Resolve a checkpoint id to its hosting env. A ``<family>:custom`` id
    looks up the ``:custom`` entries, everything else the canonical ids (the
    two can't collide: canonical ids may not contain ``:``).

    Deliberately does not check that the env is built — resolution is also
    used for metadata lookups; callers that spawn a worker check before
    spawning.

    Raises CheckpointNotFoundError when nothing matches.
    """
    if is_custom_checkpoint(checkpoint_id):
        return _resolve_custom(root, checkpoint_id)
    env_name, _ = find_env_for_checkpoint(root, checkpoint_id)
    return ResolvedCheckpoint(checkpoint=checkpoint_id, env_name=env_name)


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
