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


def read_layout_version(root: Path) -> int | None:
    """Return the root's recorded layout version, or None if unrecorded.

    None means either an empty/new root or a legacy install from before the
    marker existed — both are layout 1 in practice. A corrupt marker also
    reads as None: a broken metadata file must not brick an otherwise
    working install.
    """
    marker = Path(root) / MARKER_NAME
    try:
        version = json.loads(marker.read_text()).get("layout_version")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return version if isinstance(version, int) else None


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


def write_layout_marker(root: Path) -> None:
    """Record the current layout version in {root}/layout.json.

    Called from maintainer commands that write the root anyway (install,
    init). No-op when the recorded version is already current, so repeated
    installs don't churn the file. Atomic write, mode honoring the process
    umask — same recipe as save_manifest.
    """
    from . import __version__
    from .manifest import now_iso

    root = Path(root)
    if read_layout_version(root) == LAYOUT_VERSION:
        return

    data = {
        "layout_version": LAYOUT_VERSION,
        "written_by": f"rootstock {__version__}",
        "written_at": now_iso(),
    }

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
