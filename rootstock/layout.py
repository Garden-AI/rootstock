"""On-disk layout versioning for a rootstock install root.

The directory layout under ``{root}`` — ``environments/`` source files,
``envs/<name>/`` venvs with ``env_source.py`` inside, ``.python/``
interpreters, ``cache/``, ``home/`` — is an implicit contract that every
future client must be able to read for as long as installs live on cluster
filesystems. ``{root}/layout.json`` makes that contract explicit: a client
that meets a layout newer than it understands can say so and stop, instead
of misreading the tree by accident.

Rules for future changes:
- Additive changes (new files/dirs old clients ignore) do NOT bump
  LAYOUT_VERSION.
- Changes that move or reshape anything an old client reads DO bump it,
  and must come with reader code for the old layout (mirroring the manifest
  schema-migration policy).

A root without a marker is a legacy (pre-marker) install of layout 1;
maintainers' state-changing commands backfill the marker.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

LAYOUT_VERSION = 1
MARKER_NAME = "layout.json"


def _read_marker(root: Path) -> dict:
    """Best-effort read of {root}/layout.json; {} when absent or corrupt."""
    marker = Path(root) / MARKER_NAME
    try:
        data = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_layout_version(root: Path) -> int | None:
    """Return the root's recorded layout version, or None if unrecorded.

    None means either an empty/new root or a legacy install from before the
    marker existed — both are layout 1 in practice. A corrupt marker also
    reads as None: a broken metadata file must not brick an otherwise
    working install.
    """
    version = _read_marker(root).get("layout_version")
    return version if isinstance(version, int) else None


def read_declared_cache_root(root: Path) -> Path | None:
    """Return the cache root this install declares for itself, if any.

    The declaration lives in {root}/layout.json — it is layout information:
    where the cache/home half of the install is, which may be a different
    filesystem than the install root (e.g. Perlmutter: code on CFS, cache on
    PSCRATCH). Installs written before the declaration existed return None.
    """
    declared = _read_marker(root).get("cache_root")
    return Path(declared) if isinstance(declared, str) and declared else None


def read_declared_stage_dir(root: Path) -> str | None:
    """Return the node-local staging base this install declares, if any.

    The declaration lives in {root}/layout.json for the same reason
    ``cache_root`` does: a registry baked into pinned clients goes stale,
    while the install's own declaration travels with the install. The value
    is kept as the raw string — it may contain environment variables
    (``$SLURM_TMPDIR``, ``$TMPDIR``) that must expand on the *node* doing
    the staging, not wherever the maintainer ran init.
    """
    declared = _read_marker(root).get("stage_dir")
    return declared if isinstance(declared, str) and declared else None


def resolve_cache_root(root: Path, explicit: Path | str | None = None) -> Path:
    """Resolve the model-weight cache root for an install root.

    Resolution order:
      1. an explicit override from the caller (CLI flag / calculator kwarg),
      2. the install's own declaration in {root}/layout.json,
      3. legacy fallback for installs predating the declaration: the cluster
         registry's entry for this root (reverse lookup),
      4. the install root itself.

    Every entry point resolves through here, so the CLI and the calculator
    can't disagree about where an install's cache lives.
    """
    root = Path(root)
    if explicit is not None:
        return Path(explicit)

    declared = read_declared_cache_root(root)
    if declared is not None:
        return declared

    from .clusters import get_cluster, get_cluster_for_root

    cluster_name = get_cluster_for_root(root)
    if cluster_name is not None:
        return get_cluster(cluster_name).resolved_cache_root
    return root


def ensure_layout_compatible(root: Path) -> None:
    """Raise RuntimeError if the root's layout is newer than this client.

    Read-only; safe for non-maintainer users on shared installs.
    """
    version = read_layout_version(root)
    if version is not None and version > LAYOUT_VERSION:
        raise RuntimeError(
            f"install at {root} uses on-disk layout version {version}, but "
            f"this rootstock only understands up to {LAYOUT_VERSION}. It was "
            f"written by a newer rootstock — upgrade this client "
            f"(`pip install -U rootstock`)."
        )


def write_layout_marker(
    root: Path,
    cache_root: Path | str | None = None,
    stage_dir: str | None = None,
) -> None:
    """Record the layout version — and the install's cache root and staging
    base — in {root}/layout.json.

    Called from maintainer commands that write the root anyway (install,
    init). ``cache_root`` records where this install keeps its model-weight
    cache; ``stage_dir`` records the node-local base worker spawns may stage
    envs to (#180). When omitted, existing declarations are preserved; pass
    ``stage_dir=""`` to remove one. No-op when nothing would change, so
    repeated installs don't churn the file. Atomic write, mode honoring the
    process umask — same recipe as save_manifest.
    """
    from . import __version__
    from .manifest import now_iso

    root = Path(root)
    declared = str(cache_root) if cache_root is not None else None
    if declared is None:
        existing = read_declared_cache_root(root)
        declared = str(existing) if existing is not None else None

    declared_stage = stage_dir if stage_dir is not None else read_declared_stage_dir(root)
    declared_stage = declared_stage or None  # "" clears the declaration

    current = _read_marker(root)
    if (
        current.get("layout_version") == LAYOUT_VERSION
        and current.get("cache_root") == declared
        and current.get("stage_dir") == declared_stage
    ):
        return

    data = {
        "layout_version": LAYOUT_VERSION,
        "written_by": f"rootstock {__version__}",
        "written_at": now_iso(),
    }
    if declared is not None:
        data["cache_root"] = declared
    if declared_stage is not None:
        data["stage_dir"] = declared_stage

    root.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=root, suffix=".json")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        umask_value = os.umask(0)
        os.umask(umask_value)
        os.chmod(temp_path, 0o666 & ~umask_value)
        Path(temp_path).rename(root / MARKER_NAME)
    except Exception:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass
        raise
