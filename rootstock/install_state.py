"""
Unified reader for the state of a rootstock installation.

The filesystem is the source of truth for *what is installed*: which envs
are built, their source files, hashes, and declared checkpoints. The
manifest is the source of truth only for what cannot be derived from disk:
fetch/verify timestamps, build times, maintainer, and cluster identity.

Everything that renders or publishes install state — the ``status`` human
view, ``status --json``, and the manifest refresh behind ``manifest push`` —
reads through :func:`read_install_state` so the views cannot disagree about
which environments exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import (
    list_built_environments,
    list_environments,
    parse_checkpoints_dict,
)
from .manifest import (
    EnvironmentInfo,
    Manifest,
    compute_source_hash,
    load_manifest,
)

# Sentinel distinguishing "load the manifest for me" from an explicit
# manifest (including an explicit None).
# Sentinel meaning "load the manifest from disk"; typed Any so it can stand in
# for the Manifest | None parameter it defaults.
_LOAD: Any = object()


@dataclass
class EnvState:
    """Merged view of one built environment.

    Disk-derived fields describe what is actually installed. ``record`` is
    the manifest's metadata for the env and may be absent (manifest lagging
    the filesystem) or stale (``source_hash_drifted``).
    """

    name: str
    path: Path
    source_file: Path | None  # envs/<name>/env_source.py, if present
    source_hash: str | None
    lock_hash: str | None
    # CHECKPOINTS parsed from env_source.py; None when the source file is
    # missing or its CHECKPOINTS dict is malformed.
    declared_checkpoints: dict[str, str] | None
    record: EnvironmentInfo | None

    @property
    def source_hash_drifted(self) -> bool:
        """True when env_source.py on disk no longer matches the manifest —
        e.g. an in-place hotfix that hasn't been re-recorded yet."""
        return (
            self.record is not None
            and self.source_hash is not None
            and self.record.source_hash != self.source_hash
        )


@dataclass
class InstallState:
    root: Path
    sources: list[tuple[str, Path]]  # environments/*.py source files
    envs: dict[str, EnvState]  # every built env under envs/
    manifest: Manifest | None

    @property
    def manifest_only_envs(self) -> dict[str, EnvironmentInfo]:
        """Manifest environment records with no built env behind them."""
        if self.manifest is None:
            return {}
        return {
            name: env for name, env in self.manifest.environments.items() if name not in self.envs
        }


def read_install_state(root: Path | str, manifest: Manifest | None = _LOAD) -> InstallState:
    """Read the merged filesystem + manifest state of an installation.

    Args:
        root: Rootstock install root.
        manifest: Pass an already-loaded (or freshly created) ``Manifest`` to
            join against instead of loading from disk — callers that are
            about to mutate a manifest must pass their own instance so the
            per-env ``record`` fields alias it.
    """
    root = Path(root)
    if manifest is _LOAD:
        manifest = load_manifest(root)

    envs: dict[str, EnvState] = {}
    for name, path in list_built_environments(root):
        source_file: Path | None = path / "env_source.py"
        source_hash = None
        declared = None
        if source_file.exists():
            source_hash = compute_source_hash(source_file)
            try:
                declared = parse_checkpoints_dict(source_file)
            except ValueError:
                declared = None
        else:
            source_file = None

        lock_file = path / "env_source.py.lock"
        lock_hash = compute_source_hash(lock_file) if lock_file.exists() else None

        envs[name] = EnvState(
            name=name,
            path=path,
            source_file=source_file,
            source_hash=source_hash,
            lock_hash=lock_hash,
            declared_checkpoints=declared,
            record=manifest.environments.get(name) if manifest else None,
        )

    return InstallState(
        root=root,
        sources=list_environments(root),
        envs=envs,
        manifest=manifest,
    )
