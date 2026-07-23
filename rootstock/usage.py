"""Anonymous per-session usage records, spooled to the shared install.

Design (issue #47): the calculator side can't phone home — compute nodes are
often airgapped — and per-user homes can't be aggregated, so the only place
every user's process and the maintainer's collector can both reach is the
shared filesystem. Each worker session therefore drops one small JSON record
into ``{cache_root}/usage/``; a maintainer-side collector aggregates the
spool from a login node.

The spool directory is *provisioned*, never created here: ``{cache_root}`` is
maintainer-writable only, and a missing ``usage/`` is how an install opts out
of collection entirely. When the directory exists it must be world-writable
(``setup-perms`` creates it ``1777``, /tmp-style) so any user's session can
drop a record.

Records are one file per session, named uniquely and created with
``O_CREAT | O_EXCL`` — concurrent sessions across nodes never share a file,
so no cross-node locking is needed (a single shared append log on GPFS or
Lustre is the contention problem ``manifest_lock`` exists to avoid).

Telemetry must never break a calculation: every failure mode here — spool
dir missing or unwritable, cache root on a read-only mount, name collision —
is swallowed and logged at DEBUG. Users opt out with
``ROOTSTOCK_DISABLE_USAGE_STATS=1``.

The records are anonymous: no username, no job id. Local (user-registered)
checkpoint ids are user-chosen names and may carry meaning, so they are
recorded as ``(local)``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from pathlib import Path

logger = logging.getLogger("rootstock.usage")

USAGE_DIR_NAME = "usage"
RECORD_SCHEMA_VERSION = 1
DISABLE_ENV_VAR = "ROOTSTOCK_DISABLE_USAGE_STATS"

# Recorded in place of a local checkpoint's id: ids registered with
# `rootstock add-local` are user-chosen and may leak what someone is working
# on, which anonymous stats must not do.
LOCAL_CHECKPOINT_LABEL = "(local)"


def usage_dir(cache_root: Path | str) -> Path:
    """The spool directory for an install's cache root."""
    return Path(cache_root) / USAGE_DIR_NAME


def usage_disabled() -> bool:
    """True when the user opted out via ROOTSTOCK_DISABLE_USAGE_STATS."""
    value = os.environ.get(DISABLE_ENV_VAR, "").strip().lower()
    return value not in ("", "0", "false")


def record_session(
    *,
    root: Path | str,
    cache_root: Path | str,
    env_name: str,
    checkpoint: str,
    is_local: bool,
    device: str,
    started_at: str,
    duration_s: float,
    n_calculations: int,
) -> Path | None:
    """Best-effort: write one usage record for a finished worker session.

    Returns the record path, or None when nothing was written — opted out,
    spool directory not provisioned, or any error at all. Never raises.

    Args:
        root: Install root (identifies "where" alongside the cluster name,
            which is reverse-looked-up from it).
        cache_root: Resolved cache root; the spool lives under it.
        env_name: Pre-built environment that hosted the session.
        checkpoint: Canonical checkpoint id. Replaced with ``(local)`` when
            ``is_local`` — user-chosen ids must not leak into the spool.
        is_local: Whether the checkpoint was a user-registered weights file.
        device: Device string the worker ran on.
        started_at: ISO 8601 UTC timestamp of worker connect.
        duration_s: Wall-clock session length in seconds.
        n_calculations: Completed force calls served by the session.
    """
    try:
        if usage_disabled():
            return None

        spool = usage_dir(cache_root)
        if not spool.is_dir():
            # Unprovisioned install: collection is off. Never create the
            # directory from here — a user-created spool would be owned by
            # whoever ran first and unwritable to everyone else.
            return None

        from . import __version__
        from .clusters import get_cluster_for_root

        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "started_at": started_at,
            "duration_s": round(duration_s, 1),
            "cluster": get_cluster_for_root(root),
            "root": str(root),
            "env": env_name,
            "checkpoint": LOCAL_CHECKPOINT_LABEL if is_local else checkpoint,
            "device": device,
            "rootstock_version": __version__,
            "n_calculations": n_calculations,
        }

        # Timestamp + host + pid + random suffix: unique without coordination.
        stamp = started_at.replace(":", "").replace("+", "p")
        name = f"{stamp}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}.json"
        path = spool / name

        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(record, f)
            f.write("\n")
        return path
    except Exception:
        logger.debug("usage record skipped", exc_info=True)
        return None
