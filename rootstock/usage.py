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

The records carry no raw username and no job id. Local (user-registered)
checkpoint ids are user-chosen names and may carry meaning, so they are
recorded as ``(local)``. For distinct-user counts, each record carries a
salted, truncated hash of the username (see ``_user_hash``): pseudonymous
rather than strictly anonymous — but reversing it requires the per-install
salt, which never leaves the cluster, and on-cluster the raw record files
already reveal their writer through file ownership anyway. The hashes exist
so the collector can count unique users; anything pushed off-cluster should
carry only the counts, never the hashes.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import socket
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("rootstock.usage")

USAGE_DIR_NAME = "usage"
RECORD_SCHEMA_VERSION = 1
DISABLE_ENV_VAR = "ROOTSTOCK_DISABLE_USAGE_STATS"
SALT_FILE_NAME = "salt"

# Recorded in place of a local checkpoint's id: ids registered with
# `rootstock add-local` are user-chosen and may leak what someone is working
# on, which anonymous stats must not do.
LOCAL_CHECKPOINT_LABEL = "(local)"


def usage_dir(cache_root: Path | str) -> Path:
    """The spool directory for an install's cache root."""
    return Path(cache_root) / USAGE_DIR_NAME


def _user_hash(spool: Path) -> str | None:
    """Salted, truncated hash of the username, for distinct-user counting.

    A bare hash of a username is reversible by enumeration — usernames are
    low-entropy and public on a cluster — so the hash is salted with a random
    per-install value living at ``{spool}/salt``. The salt is created lazily
    by whichever session writes first (O_EXCL, so exactly one writer wins)
    and must be world-readable, since every writer needs it; that means the
    hashes are reversible *on* the cluster (where record-file ownership
    already names the writer) but opaque anywhere the salt doesn't go.

    Returns None when anything fails — the record is then simply written
    without a user field.
    """
    try:
        salt_path = spool / SALT_FILE_NAME
        try:
            salt = salt_path.read_bytes()
        except FileNotFoundError:
            salt = uuid.uuid4().bytes + uuid.uuid4().bytes
            try:
                fd = os.open(salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                with os.fdopen(fd, "wb") as f:
                    f.write(salt)
            except FileExistsError:
                salt = salt_path.read_bytes()
        if not salt:
            return None
        username = getpass.getuser()
        return hashlib.sha256(salt + username.encode("utf-8")).hexdigest()[:16]
    except Exception:
        logger.debug("user hash skipped", exc_info=True)
        return None


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
    client: str,
    started_at: str,
    duration_s: float,
    n_calculations: int | None,
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
            ``<family>:custom`` ids are exempt and recorded verbatim: the
            marker already self-flags the run as user weights, and the id is
            env-declared, not user-chosen.
        is_local: Whether the checkpoint was a user-registered weights file.
        device: Device string the worker ran on.
        client: Which entry point ran the session ("calculator", "serve").
        started_at: ISO 8601 UTC timestamp of session start (worker connect,
            or worker spawn when the connect isn't observable).
        duration_s: Wall-clock session length in seconds.
        n_calculations: Completed force calls served by the session, or None
            when the entry point can't count them (serve's parent process
            never sees the i-PI traffic).
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
        from .environment import is_custom_checkpoint

        # <family>:custom is non-identifying (no user path; the id is
        # env-declared, not user-chosen) and self-flags the run as user
        # weights — record it verbatim so the fine-tuned *family* stays
        # visible in the stats.
        mask = is_local and not is_custom_checkpoint(checkpoint)

        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "started_at": started_at,
            "duration_s": round(duration_s, 1),
            "cluster": get_cluster_for_root(root),
            "root": str(root),
            "env": env_name,
            "checkpoint": LOCAL_CHECKPOINT_LABEL if mask else checkpoint,
            "device": device,
            "client": client,
            "rootstock_version": __version__,
            "n_calculations": n_calculations,
            "user": _user_hash(spool),
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


# --------------------------------------------------------------------------- #
# Collector: aggregate the spool, compact raw records into monthly rollups
# --------------------------------------------------------------------------- #

ROLLUP_PREFIX = "rollup-"
ROLLUP_SCHEMA_VERSION = 1

# One aggregated row per distinct combination of these; every other record
# field is either summed (n_calculations, duration_s) or counted (sessions).
# client is a key so entry-point adoption (calculator vs serve/LAMMPS) stays
# visible after compaction — rollups discard everything not keyed or summed.
_KEY_FIELDS = ("month", "cluster", "env", "checkpoint", "device", "client")
_REQUIRED_RECORD_FIELDS = ("started_at", "env", "checkpoint", "device", "n_calculations")


@dataclass
class SpoolSummary:
    """Aggregated view of a spool: rollup rows merged with raw records.

    Each row carries ``unique_users``, a count derived from the salted user
    hashes; the hashes themselves stay in the spool (raw records and rollup
    files need them for exact set-union merging) and are never part of a
    summary — counts are what may leave the cluster.
    """

    rows: list[dict]
    raw_files: int  # raw records aggregated (report) or compacted (compact)
    skipped: int  # unreadable/malformed/newer-schema files left in place
    kept: int = 0  # compact only: undeletable raw files left for the owner
    unique_users: int = 0  # distinct user hashes across every row


def _record_key(record: dict) -> tuple:
    month = str(record["started_at"])[:7]
    return (month,) + tuple(record.get(f) for f in _KEY_FIELDS[1:])


def _merge_record(rows: dict[tuple, dict], key: tuple, sessions, calls, seconds, users) -> None:
    row = rows.setdefault(
        key,
        dict(zip(_KEY_FIELDS, key))
        | {"sessions": 0, "n_calculations": 0, "duration_s": 0.0, "users": set()},
    )
    row["sessions"] += sessions
    row["n_calculations"] += calls
    row["duration_s"] = round(row["duration_s"] + seconds, 1)
    row["users"].update(u for u in users if isinstance(u, str))


def _load_json(path: Path) -> dict | None:
    """A record/rollup this collector understands, or None (count as skipped)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("schema_version")
    if not isinstance(version, int) or version > RECORD_SCHEMA_VERSION:
        return None
    return data


def _spool_files(spool: Path) -> tuple[list[Path], list[Path]]:
    """(raw record files, rollup files) currently in the spool."""
    raw: list[Path] = []
    rollups: list[Path] = []
    for path in sorted(spool.glob("*.json")):
        (rollups if path.name.startswith(ROLLUP_PREFIX) else raw).append(path)
    return raw, rollups


def _load_rollup_rows(rollup_paths: list[Path], rows: dict[tuple, dict]) -> int:
    """Merge previously compacted rows in; returns how many files were skipped."""
    skipped = 0
    for path in rollup_paths:
        data = _load_json(path)
        if data is None or not isinstance(data.get("rows"), list):
            skipped += 1
            continue
        for row in data["rows"]:
            try:
                key = tuple(row[f] for f in _KEY_FIELDS)
                _merge_record(
                    rows,
                    key,
                    row["sessions"],
                    row["n_calculations"],
                    row["duration_s"],
                    row.get("users", []),
                )
            except (KeyError, TypeError):
                skipped += 1
                break
    return skipped


def _record_users(record: dict) -> list[str]:
    user = record.get("user")
    return [user] if isinstance(user, str) else []


def summarize_spool(cache_root: Path | str) -> SpoolSummary | None:
    """Aggregate everything in the spool (rollups + raw records), read-only.

    Returns None when the install has no spool at all (collection is off).
    """
    spool = usage_dir(cache_root)
    if not spool.is_dir():
        return None

    raw_paths, rollup_paths = _spool_files(spool)
    rows: dict[tuple, dict] = {}
    skipped = _load_rollup_rows(rollup_paths, rows)

    raw_count = 0
    for path in raw_paths:
        record = _load_json(path)
        if record is None or any(f not in record for f in _REQUIRED_RECORD_FIELDS):
            skipped += 1
            continue
        _merge_record(
            rows,
            _record_key(record),
            1,
            # serve records carry n_calculations=null (the parent process
            # never sees the i-PI traffic): count the session, sum nothing.
            record["n_calculations"] or 0,
            float(record.get("duration_s") or 0.0),
            _record_users(record),
        )
        raw_count += 1

    ordered = sorted(rows.values(), key=lambda r: tuple(str(r[f]) for f in _KEY_FIELDS))
    # Summaries expose counts, never the hashes themselves.
    all_users: set[str] = set()
    for row in ordered:
        users = row.pop("users")
        all_users |= users
        row["unique_users"] = len(users)
    return SpoolSummary(
        rows=ordered, raw_files=raw_count, skipped=skipped, unique_users=len(all_users)
    )


def compact_spool(cache_root: Path | str) -> SpoolSummary | None:
    """Fold raw records into per-month ``rollup-YYYY-MM.json`` files.

    Each raw file is deleted *before* its counts are folded in; a file the
    caller can't delete (the spool is sticky, so only a record's owner or the
    spool's owner can) is left alone and reported via ``kept`` — merging it
    anyway would double-count it on the next run. The flip side: a crash
    between delete and rollup write can lose those records, which is an
    acceptable trade for telemetry.

    Returns None when the install has no spool. Idempotent: re-running is a
    no-op until new records arrive.
    """
    spool = usage_dir(cache_root)
    if not spool.is_dir():
        return None

    raw_paths, rollup_paths = _spool_files(spool)

    skipped = 0
    kept = 0
    compacted = 0
    new_rows: dict[tuple, dict] = {}
    for path in raw_paths:
        record = _load_json(path)
        if record is None or any(f not in record for f in _REQUIRED_RECORD_FIELDS):
            skipped += 1
            continue
        try:
            path.unlink()
        except OSError:
            kept += 1
            continue
        _merge_record(
            new_rows,
            _record_key(record),
            1,
            record["n_calculations"] or 0,  # null for serve records
            float(record.get("duration_s") or 0.0),
            _record_users(record),
        )
        compacted += 1

    # Fold the new counts into each affected month's existing rollup.
    months = {key[0] for key in new_rows}
    for month in sorted(months):
        rollup_path = spool / f"{ROLLUP_PREFIX}{month}.json"
        month_rows: dict[tuple, dict] = {}
        if rollup_path.exists():
            skipped += _load_rollup_rows([rollup_path], month_rows)
        for key, row in new_rows.items():
            if key[0] == month:
                _merge_record(
                    month_rows,
                    key,
                    row["sessions"],
                    row["n_calculations"],
                    row["duration_s"],
                    row["users"],
                )
        ordered = sorted(month_rows.values(), key=lambda r: tuple(str(r[f]) for f in _KEY_FIELDS))
        # Rollups keep the hashes (sorted for stable diffs): exact set-union
        # merging on later compactions needs them. They stay in the spool.
        serial = [dict(row, users=sorted(row["users"])) for row in ordered]
        payload = {"schema_version": ROLLUP_SCHEMA_VERSION, "month": month, "rows": serial}
        # Atomic replace, same recipe as save_manifest: a reader (or a
        # concurrent report) never sees a half-written rollup.
        fd, temp_path = tempfile.mkstemp(dir=spool, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            os.chmod(temp_path, 0o644)
            Path(temp_path).rename(rollup_path)
        except Exception:
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
            raise

    summary = summarize_spool(cache_root)
    summary.raw_files = compacted
    summary.skipped = skipped
    summary.kept = kept
    return summary
