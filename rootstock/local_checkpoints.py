"""
Per-user registry of local checkpoints (user-supplied weights files).

A shared install is world-readable but maintainer-writable, so a user who
fine-tunes a model cannot register the weights in the install's manifest or
env sources. Instead, local checkpoints live in a per-user JSON registry
(``~/.config/rootstock/local-checkpoints.json``) keyed by install root. Each
entry binds a user-chosen checkpoint id to an installed env that declares the
``setup_from_path`` hook, plus the weights path, its hash, and verification
state.

Resolution (:func:`resolve_checkpoint`) consults the env-declared canonical
ids first, then this registry — so a registered id works everywhere a
canonical id does (calculator, serve, benchmark) without any shared-root
writes.

Locking: none, deliberately. The registry is a private per-user file; the
only plausible concurrent writers are the same user's own processes (an
``add-local`` racing a nightly ``smoke-test``). Atomic replace prevents
corruption, and a lost update costs at most one re-verification, not shared
state. Every mutator loads fresh, mutates, and saves immediately — no
registry object is held across long-running work.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DEFAULT_CONFIG_DIR
from .environment import (
    CheckpointNotFoundError,
    declares_setup_from_path,
    find_env_for_checkpoint,
    list_declared_checkpoints,
)
from .exceptions import RootstockError
from .manifest import now_iso

LOCAL_REGISTRY_SCHEMA_VERSION = 1
DEFAULT_LOCAL_REGISTRY_FILE = DEFAULT_CONFIG_DIR / "local-checkpoints.json"

# Keys setup_from_path receives positionally / at the top level. "path" is
# reserved too: it's setup_from_path's first parameter, so a registered
# path= kwarg would TypeError with "multiple values for argument".
RESERVED_SETUP_KWARGS = frozenset({"checkpoint", "device", "path"})

_HASH_CHUNK_SIZE = 1024 * 1024  # weights files are GBs; read in 1 MiB chunks


class LocalCheckpointError(RootstockError, RuntimeError):
    """A local-checkpoint registry operation failed. Messages are
    user-presentable diagnoses, not tracebacks."""


@dataclass
class LocalCheckpointEntry:
    """One registered local checkpoint, bound to an install root."""

    env: str
    path: str  # absolute path to the weights file (user's own storage)
    sha256: str  # "sha256:<hex>", matching the manifest's source_hash style
    size: int  # st_size at registration; cheap staleness signal
    setup_kwargs: dict = field(default_factory=dict)
    registered_at: str = ""  # ISO 8601
    verified_at: str | None = None
    verified_device: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LocalCheckpointEntry:
        return cls(
            env=data["env"],
            path=data["path"],
            sha256=data["sha256"],
            size=data["size"],
            setup_kwargs=data.get("setup_kwargs") or {},
            registered_at=data.get("registered_at", ""),
            verified_at=data.get("verified_at"),
            verified_device=data.get("verified_device"),
            last_error=data.get("last_error"),
        )


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Where a checkpoint id points: hosting env, plus the weights path and
    registered default setup kwargs when the id is a local checkpoint."""

    checkpoint: str
    env_name: str
    path: str | None = None  # None => canonical id
    setup_kwargs: dict = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return self.path is not None


def _root_key(root: Path | str) -> str:
    """Registry key for an install root. Resolved so the same install reached
    via symlinks or relative paths lands on one entry set."""
    return str(Path(root).resolve())


def _registry_path(registry_path: Path | str | None) -> Path:
    # Resolved at call time (not bound as a default) so tests can point the
    # module-level default somewhere else.
    if registry_path is not None:
        return Path(registry_path)
    return DEFAULT_LOCAL_REGISTRY_FILE


def hash_weights_file(path: Path) -> tuple[str, int]:
    """Chunked sha256 of a weights file. Returns ("sha256:<hex>", size)."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return f"sha256:{digest.hexdigest()}", size


def load_local_registry(
    registry_path: Path | str | None = None,
) -> dict[str, dict[str, LocalCheckpointEntry]]:
    """
    Load the registry as ``{root_key: {checkpoint_id: entry}}``.

    A missing file is an empty registry. A file that exists but can't be
    parsed raises — unlike the manifest, the recovery hint is cheap because
    only this user's registrations are at stake.
    """
    path = _registry_path(registry_path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
        version = data.get("schema_version")
        if not isinstance(version, int):
            raise ValueError(f"invalid schema_version={version!r}")
        if version > LOCAL_REGISTRY_SCHEMA_VERSION:
            raise LocalCheckpointError(
                f"local-checkpoint registry at {path} is schema_version="
                f"{version}, but this rootstock only understands up to "
                f"{LOCAL_REGISTRY_SCHEMA_VERSION}. It was written by a newer "
                f"rootstock — upgrade this client."
            )
        return {
            root_key: {
                ckpt_id: LocalCheckpointEntry.from_dict(entry) for ckpt_id, entry in entries.items()
            }
            for root_key, entries in data.get("roots", {}).items()
        }
    except LocalCheckpointError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LocalCheckpointError(
            f"local-checkpoint registry at {path} is corrupted "
            f"({type(exc).__name__}: {exc}). Fix the file, or delete it to "
            f"start fresh — it only holds your own registrations; weights "
            f"files are never stored in it."
        ) from exc


def save_local_registry(
    roots: dict[str, dict[str, LocalCheckpointEntry]],
    registry_path: Path | str | None = None,
) -> None:
    """Atomic write (temp + rename). mkstemp's 0600 is correct here — this is
    a private per-user file, unlike the shared manifest."""
    path = _registry_path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "schema_version": LOCAL_REGISTRY_SCHEMA_VERSION,
        "roots": {
            root_key: {ckpt_id: entry.to_dict() for ckpt_id, entry in entries.items()}
            for root_key, entries in roots.items()
        },
    }

    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(temp_path).rename(path)
    except Exception:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass
        raise


def local_checkpoints_for_root(
    root: Path | str,
    registry_path: Path | str | None = None,
) -> dict[str, LocalCheckpointEntry]:
    """Registered local checkpoints for one install root (empty dict if none)."""
    return load_local_registry(registry_path).get(_root_key(root), {})


def register_local_checkpoint(
    root: Path | str,
    checkpoint_id: str,
    env_name: str,
    weights_path: Path | str,
    setup_kwargs: dict | None = None,
    registry_path: Path | str | None = None,
) -> LocalCheckpointEntry:
    """
    Validate and record a local checkpoint. Idempotent: re-registering an
    existing id overwrites the entry and resets its verification state.

    Raises LocalCheckpointError for every validation failure; the messages
    name the fix.
    """
    root = Path(root)
    weights_path = Path(weights_path).expanduser().resolve()
    setup_kwargs = dict(setup_kwargs or {})

    if not weights_path.is_file():
        raise LocalCheckpointError(
            f"weights file not found (or not a regular file): {weights_path}"
        )
    if not os.access(weights_path, os.R_OK):
        raise LocalCheckpointError(f"weights file is not readable: {weights_path}")

    env_dir = root / "envs" / env_name
    env_source = env_dir / "env_source.py"
    if not (env_dir / "bin" / "python").exists() or not env_source.exists():
        built = (
            sorted(
                p.name
                for p in (root / "envs").iterdir()
                if p.is_dir() and (p / "bin" / "python").exists()
            )
            if (root / "envs").exists()
            else []
        )
        raise LocalCheckpointError(
            f"env '{env_name}' is not built at {root}. "
            f"Built envs: {', '.join(built) if built else '(none)'}."
        )

    if not declares_setup_from_path(env_source):
        supporting = sorted(
            env_dir.name
            for env_dir in (root / "envs").iterdir()
            if (env_dir / "env_source.py").exists()
            and declares_setup_from_path(env_dir / "env_source.py")
        )
        raise LocalCheckpointError(
            f"env '{env_name}' does not declare setup_from_path(path, device, "
            f"**kwargs), which is required to load local checkpoints. "
            f"Envs at {root} that support local checkpoints: "
            f"{', '.join(supporting) if supporting else '(none)'}. "
            f"Ask the install maintainer to add setup_from_path to the env "
            f"source — see docs/environments.md."
        )

    for owner_env, declared in list_declared_checkpoints(root).items():
        if checkpoint_id in declared:
            raise LocalCheckpointError(
                f"id '{checkpoint_id}' is already a canonical checkpoint of "
                f"env '{owner_env}' at {root}; choose a different --id."
            )

    reserved = RESERVED_SETUP_KWARGS & setup_kwargs.keys()
    if reserved:
        raise LocalCheckpointError(
            f"setup_kwargs cannot contain reserved keys {sorted(reserved)}; "
            f"they are passed to setup_from_path at the top level."
        )

    sha256, size = hash_weights_file(weights_path)
    entry = LocalCheckpointEntry(
        env=env_name,
        path=str(weights_path),
        sha256=sha256,
        size=size,
        setup_kwargs=setup_kwargs,
        registered_at=now_iso(),
    )

    roots = load_local_registry(registry_path)
    roots.setdefault(_root_key(root), {})[checkpoint_id] = entry
    save_local_registry(roots, registry_path)
    return entry


def remove_local_checkpoint(
    root: Path | str,
    checkpoint_id: str,
    registry_path: Path | str | None = None,
) -> LocalCheckpointEntry:
    """
    Delete a registry entry and return it. Never touches the weights file —
    the registry records it, it doesn't own it.
    """
    roots = load_local_registry(registry_path)
    key = _root_key(root)
    entries = roots.get(key, {})
    if checkpoint_id not in entries:
        if entries:
            listing = ", ".join(sorted(entries))
            detail = f"Registered local checkpoints for {key}: {listing}."
        else:
            detail = f"No local checkpoints are registered for {key}."
        raise LocalCheckpointError(f"no local checkpoint '{checkpoint_id}' is registered. {detail}")

    entry = entries.pop(checkpoint_id)
    if not entries:
        del roots[key]
    save_local_registry(roots, registry_path)
    return entry


def record_local_verification(
    root: Path | str,
    checkpoint_id: str,
    *,
    ok: bool,
    device: str,
    error: str | None = None,
    registry_path: Path | str | None = None,
) -> None:
    """Write a verification outcome back to the registry. A checkpoint
    removed since the caller last looked is skipped silently — the outcome
    has nothing to attach to."""
    roots = load_local_registry(registry_path)
    entry = roots.get(_root_key(root), {}).get(checkpoint_id)
    if entry is None:
        return
    if ok:
        entry.verified_at = now_iso()
        entry.verified_device = device
        entry.last_error = None
    else:
        entry.last_error = error
    save_local_registry(roots, registry_path)


def resolve_checkpoint(
    root: Path | str,
    checkpoint_id: str,
    registry_path: Path | str | None = None,
) -> ResolvedCheckpoint:
    """
    Resolve a checkpoint id to its hosting env, checking env-declared
    canonical ids first, then the user's local-checkpoint registry.

    Canonical ids win: registration rejects collisions in the other
    direction, but an env installed *after* a local registration can
    introduce one — the canonical id is authoritative and `status` surfaces
    the shadowing.

    Deliberately does not check that the env is built or the weights file
    still exists — resolution is also used for metadata lookups; callers
    that spawn a worker check before spawning.

    Raises CheckpointNotFoundError when neither namespace has the id.
    """
    for env_name, ckpts in list_declared_checkpoints(root).items():
        if checkpoint_id in ckpts:
            return ResolvedCheckpoint(checkpoint=checkpoint_id, env_name=env_name)

    try:
        local = local_checkpoints_for_root(root, registry_path)
    except LocalCheckpointError as exc:
        # A broken per-user file must not brick canonical resolution; the
        # mutating commands (add-local/remove-local) still fail loudly.
        print(f"Warning: ignoring local-checkpoint registry: {exc}", file=sys.stderr)
        local = {}

    entry = local.get(checkpoint_id)
    if entry is not None:
        return ResolvedCheckpoint(
            checkpoint=checkpoint_id,
            env_name=entry.env,
            path=entry.path,
            setup_kwargs=dict(entry.setup_kwargs),
        )

    # Reuse find_env_for_checkpoint's rich canonical listing, appending the
    # local dimension so the error names both namespaces.
    try:
        find_env_for_checkpoint(root, checkpoint_id)
    except CheckpointNotFoundError as exc:
        extra = ""
        if local:
            extra = f"\nYour registered local checkpoints: {', '.join(sorted(local))}."
        raise CheckpointNotFoundError(
            f"{exc}{extra}\nLocal weights files can be registered with "
            f"`rootstock add-local <path> --env <env> --id <id>`."
        ) from None
    # find_env_for_checkpoint succeeding here is impossible (we already
    # walked the same declarations), but never mask it if it happens.
    raise AssertionError("resolve_checkpoint: inconsistent declared-checkpoint walk")
