"""
Manifest management for Rootstock.

The manifest tracks the state of a rootstock installation:
- Which environments are built
- Dependency versions
- Checkpoints available
- Last update time
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import UserConfig
from .exceptions import RootstockError

SCHEMA_VERSION = 7

# manifest_lock defaults. The lock is only held across a load → mutate → save
# cycle (plus the env refresh, which shells out to `uv pip list` per env), so
# holds are seconds, not minutes; the stale threshold is deliberately far
# beyond any legitimate hold.
LOCK_TIMEOUT = 120.0
LOCK_STALE_AFTER = 600.0
_LOCK_POLL_INTERVAL = 0.5


class ManifestError(RootstockError, RuntimeError):
    """The manifest exists but cannot be used: corrupt JSON, missing required
    fields, or a schema this client has no path to. Deliberately NOT treated
    as "no manifest" — a silent fresh start would let the next save overwrite
    all fetch/verify history."""


class ManifestLockTimeout(ManifestError):
    """Could not acquire the manifest lock within the timeout. Subclasses
    ManifestError so the CLI reports it as a clean diagnosis, not a
    traceback."""


def _migrate_v1_to_v2(data: dict) -> tuple[dict, str | None]:
    """v1 stored ``checkpoints: list[str]``; v2 stores a dict of CheckpointInfo.

    Each listed checkpoint becomes an empty CheckpointInfo — nothing fetched,
    nothing verified. (Port of the retired scripts/migrate_manifest_v1_to_v2.py.)
    """
    for env in data.get("environments", {}).values():
        env["checkpoints"] = {
            name: {
                "fetched_at": None,
                "verified_at": None,
                "verified_device": None,
                "last_error": None,
            }
            for name in (env.get("checkpoints") or [])
        }
    data["schema_version"] = 2
    return data, None


def _migrate_v2_to_v3(data: dict) -> tuple[dict, str | None]:
    """v3 renamed checkpoint ids to canonical form (e.g. 'mace-mp-0-small').

    v2 checkpoint keys are env-local names that no longer join against any
    env's CHECKPOINTS table, so the entries are dropped rather than carried
    with untrustworthy ids. Environments survive; model weights stay cached.
    """
    environments = data.get("environments", {})
    dropped = sum(len(env.get("checkpoints") or {}) for env in environments.values())
    for env in environments.values():
        env["checkpoints"] = {}
    data["schema_version"] = 3

    note = None
    if dropped:
        note = (
            f"dropped {dropped} checkpoint record(s): v3 switched to canonical "
            f"checkpoint ids, so v2 ids can't be trusted. Re-run "
            f"`rootstock add <checkpoint-id>` to re-verify (weights stay cached)."
        )
    return data, note


def _migrate_v3_to_v4(data: dict) -> tuple[dict, str | None]:
    """v4 dropped EnvironmentInfo.status and .error_message.

    Nothing ever wrote a status other than "ready" or set an error message,
    so removing the keys loses no information.
    """
    for env in data.get("environments", {}).values():
        env.pop("status", None)
        env.pop("error_message", None)
    data["schema_version"] = 4
    return data, None


def _migrate_v4_to_v5(data: dict) -> tuple[dict, str | None]:
    """v5 added optional weight-tracking fields to CheckpointInfo.

    ``weight_files``/``weights_recorded_at`` default to absent, so this is a
    pure version bump. The bump still matters: without it a v4 client's
    from_dict/asdict round-trip would silently drop the new fields on its
    next save — refusing loudly (migrate_manifest_data's newer-schema check)
    beats that.
    """
    data["schema_version"] = 5
    return data, None


def _migrate_v5_to_v6(data: dict) -> tuple[dict, str | None]:
    """v6 made cluster identity plural and verification per-cluster (#208).

    A shared install (sophia/polaris on Eagle) serves machines that differ in
    node image and drivers, so one flat ``verified_at`` misattributes results.
    ``cluster: str`` becomes ``clusters: list[str]``, seeded from the cluster
    registry so a root known to serve several machines starts serving them all
    without a manual re-init. Each checkpoint's flat verification fields move
    under ``verifications[<old cluster>]`` — pre-v6 results can't be
    attributed to any other cluster, so siblings start unverified. The mixed
    ``last_error`` splits by its writer's prefix: ``download:`` errors stay on
    the (shared) checkpoint, everything else was a verification error.
    EnvironmentInfo gains ``clusters`` (None = serves every cluster), picked
    up from each env source's CLUSTERS on the next refresh.
    """
    from .clusters import get_clusters_for_root

    old = data.pop("cluster", None) or "unknown"
    siblings = get_clusters_for_root(data.get("root", ""))
    data["clusters"] = [old] + [c for c in siblings if c != old]

    for env in data.get("environments", {}).values():
        env.setdefault("clusters", None)
        for ckpt in (env.get("checkpoints") or {}).values():
            verified_at = ckpt.pop("verified_at", None)
            verified_device = ckpt.pop("verified_device", None)
            last_error = ckpt.pop("last_error", None)
            is_fetch_error = last_error is not None and last_error.startswith("download:")
            ckpt["last_error"] = last_error if is_fetch_error else None
            verify_error = None if is_fetch_error else last_error
            ckpt["verifications"] = {}
            if verified_at is not None or verify_error is not None:
                ckpt["verifications"][old] = {
                    "verified_at": verified_at,
                    "verified_device": verified_device,
                    "last_error": verify_error,
                }

    data["schema_version"] = 6

    note = None
    if len(data["clusters"]) > 1:
        others = ", ".join(data["clusters"][1:])
        note = (
            f"this root also serves {others} (per the cluster registry); "
            f"prior verification history stays attributed to '{old}' and the "
            f"other cluster(s) start unverified."
        )
    return data, note


def _migrate_v6_to_v7(data: dict) -> tuple[dict, str | None]:
    """v7 added the optional per-env ``image`` record (packed env archive,
    #180).

    ``image`` defaults to absent, so this is a pure version bump — the bump
    matters for the same reason v5's did: without it a v6 client's
    from_dict/asdict round-trip would silently drop the record on its next
    save.
    """
    data["schema_version"] = 7
    return data, None


# One entry per historical schema version, upgrading one step. A schema bump
# without a migration here strands every deployed manifest of that vintage —
# add the migration in the same change as the bump.
MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,
    5: _migrate_v5_to_v6,
    6: _migrate_v6_to_v7,
}


def migrate_manifest_data(data: dict) -> tuple[dict, list[str]]:
    """Upgrade a raw manifest dict to SCHEMA_VERSION, one step at a time.

    Returns the upgraded dict and human-readable notes about lossy steps
    (empty when the manifest was already current). Raises ManifestError for
    manifests from a *newer* rootstock or with no migration path.
    """
    version = data.get("schema_version")
    if isinstance(version, str) and version.isdigit():
        version = int(version)  # v1 wrote schema_version as a string

    if not isinstance(version, int):
        raise ManifestError(
            f"manifest has invalid schema_version={data.get('schema_version')!r}; "
            f"expected an integer <= {SCHEMA_VERSION}."
        )

    if version > SCHEMA_VERSION:
        raise ManifestError(
            f"manifest is schema_version={version}, but this rootstock only "
            f"understands up to {SCHEMA_VERSION}. It was written by a newer "
            f"rootstock — upgrade this client (`pip install -U rootstock`)."
        )

    if version == SCHEMA_VERSION:
        if data["schema_version"] != version:  # normalize a coerced string
            data = {**data, "schema_version": version}
        return data, []

    # Migrations mutate freely; the caller's dict stays untouched.
    data = copy.deepcopy(data)
    data["schema_version"] = version

    notes: list[str] = []
    while version < SCHEMA_VERSION:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise ManifestError(
                f"manifest is schema_version={version} and no migration path "
                f"to {SCHEMA_VERSION} exists."
            )
        data, note = migrate(data)
        notes.append(
            f"migrated manifest schema v{version} -> v{data['schema_version']}"
            + (f": {note}" if note else "")
        )
        version = data["schema_version"]

    return data, notes


@dataclass
class Maintainer:
    """Maintainer contact information."""

    name: str
    email: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Maintainer:
        return cls(name=data["name"], email=data["email"])


@dataclass
class VerificationRecord:
    """One cluster's verification state for a checkpoint.

    Verification is per-cluster (v6, #208): a shared install serves machines
    that differ in node image and drivers, so passing on one says nothing
    about the others.
    """

    verified_at: str | None = None  # ISO 8601, set when smoke test passes
    verified_device: str | None = None  # "cuda", "cpu", etc.
    last_error: str | None = None  # most recent verify/smoke-test error here

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VerificationRecord:
        return cls(
            verified_at=data.get("verified_at"),
            verified_device=data.get("verified_device"),
            last_error=data.get("last_error"),
        )


@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint registered with an environment.

    Fetch state is shared across clusters — the weights cache is one
    filesystem however many machines mount it — while verification is
    recorded per cluster in ``verifications``.
    """

    fetched_at: str | None = None  # ISO 8601, set when download succeeds
    last_error: str | None = None  # most recent download error (shared)
    # Weight files this checkpoint's setup() actually touched, captured at
    # add/verify/smoke-test time (#177): list of {"path", "size"} dicts with
    # paths relative to the cache root (so records survive split-cache
    # installs) and sizes in bytes. None = never recorded. Granularity is
    # deliberately files, not dirs: HF hub keeps one repo dir per model
    # *family*, so dir-level records over-warm — and under a job's memory
    # cgroup, over-warming evicts the pages the worker needs.
    weight_files: list[dict] | None = None
    weights_recorded_at: str | None = None  # ISO 8601
    # Per-cluster verification state, keyed by cluster name. Absent key =
    # never verified there.
    verifications: dict[str, VerificationRecord] = field(default_factory=dict)

    def verification(self, cluster: str) -> VerificationRecord:
        """This cluster's verification state (empty record if never verified)."""
        return self.verifications.get(cluster) or VerificationRecord()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointInfo:
        return cls(
            fetched_at=data.get("fetched_at"),
            last_error=data.get("last_error"),
            weight_files=data.get("weight_files"),
            weights_recorded_at=data.get("weights_recorded_at"),
            verifications={
                cluster: VerificationRecord.from_dict(v)
                for cluster, v in (data.get("verifications") or {}).items()
            },
        )


@dataclass
class EnvironmentInfo:
    """Metadata for a single built environment.

    An entry exists iff the env is built on disk — there is no status field.
    A failed build leaves no env directory, so the manifest never has anything
    to say about it.
    """

    built_at: str  # ISO 8601 timestamp
    # "sha256:abc123...", or None when the built env has no env_source.py
    # to hash (e.g. one built by a pre-lockfile rootstock).
    source_hash: str | None
    source: str  # Full source code of the environment file
    python_requires: str  # ">=3.11"
    dependencies: dict[str, str]  # {"mace-torch": "0.3.6"}
    # sha256 of the env's uv lockfile (envs/<name>/env_source.py.lock).
    # None means the env was built without one — either by a pre-lockfile
    # rootstock or because the env defeats universal resolution — and can
    # only be re-resolved, not faithfully rebuilt.
    lock_hash: str | None = None
    # Clusters this env serves, mirrored from the source's CLUSTERS at
    # refresh time. None = serves every cluster the install does; a list
    # restricts it (a cluster-specific variant on a shared install, #208).
    clusters: list[str] | None = None
    # Packed single-image archive of this env (#180): a dict of {"path"
    # (relative to the root), "sha256" (of the archive — the staging
    # identity), "format", "compressed_bytes", "uncompressed_bytes",
    # "packed_at"}. None = never packed. The image is current only while
    # ``packed_at >= built_at`` (see :func:`image_is_current`) — a rebuild
    # without a repack must never serve a stale image.
    image: dict | None = None
    # Field order is the JSON key order (asdict): lock_hash stays ahead of
    # checkpoints to match the layout pushed manifests have always had.
    checkpoints: dict[str, CheckpointInfo] = field(default_factory=dict)

    def serves(self, cluster: str) -> bool:
        """Whether this env serves ``cluster`` (unrestricted envs serve all)."""
        return self.clusters is None or cluster in self.clusters

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EnvironmentInfo:
        checkpoints_data = data.get("checkpoints", {})
        checkpoints = {
            name: CheckpointInfo.from_dict(ckpt_data)
            for name, ckpt_data in checkpoints_data.items()
        }
        return cls(
            built_at=data["built_at"],
            source_hash=data["source_hash"],
            source=data.get("source", ""),
            python_requires=data["python_requires"],
            dependencies=data["dependencies"],
            checkpoints=checkpoints,
            lock_hash=data.get("lock_hash"),
            clusters=data.get("clusters"),
            image=data.get("image"),
        )


def is_verified(env: EnvironmentInfo, ckpt: CheckpointInfo, cluster: str) -> bool:
    """True if the checkpoint was verified on ``cluster`` after the env was
    last built."""
    verified_at = ckpt.verification(cluster).verified_at
    if verified_at is None:
        return False
    return verified_at > env.built_at  # ISO 8601 sorts lexically


def image_is_current(env: EnvironmentInfo) -> bool:
    """True if the env's packed image record describes the current build.

    ``packed_at >= built_at`` (ISO 8601 sorts lexically; ``>=`` because an
    install stamps both in the same refresh): an env rebuilt without a repack
    reads as image-less rather than serving a stale archive.
    """
    if not isinstance(env.image, dict):
        return False
    packed_at = env.image.get("packed_at")
    return isinstance(packed_at, str) and packed_at >= env.built_at


@dataclass
class Manifest:
    """Root manifest for a rootstock installation.

    One file per install, however many clusters mount it: build/fetch state
    is genuinely shared, so splitting the file would just invite drift. The
    per-cluster split happens at push time — see :func:`manifest_push_payload`.
    """

    schema_version: int
    clusters: list[str]  # every cluster this install serves; first = "home"
    root: str
    maintainer: Maintainer
    rootstock_version: str
    python_version: str
    last_updated: str  # ISO 8601 timestamp
    environments: dict[str, EnvironmentInfo] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        data, _ = migrate_manifest_data(data)

        environments = {}
        for name, env_data in data.get("environments", {}).items():
            environments[name] = EnvironmentInfo.from_dict(env_data)

        return cls(
            schema_version=data["schema_version"],
            clusters=list(data["clusters"]),
            root=data["root"],
            maintainer=Maintainer.from_dict(data["maintainer"]),
            rootstock_version=data["rootstock_version"],
            python_version=data["python_version"],
            last_updated=data["last_updated"],
            environments=environments,
        )

    def validate(self) -> tuple[bool, str]:
        """
        Validate manifest structure.

        Returns:
            (success, message) tuple
        """
        # Check required fields
        if not self.schema_version:
            return False, "Missing schema_version"
        if not self.clusters:
            return False, "Missing clusters"
        if not self.root:
            return False, "Missing root"
        if not self.maintainer.name:
            return False, "Missing maintainer name"
        if not self.maintainer.email:
            return False, "Missing maintainer email"

        return True, "OK"


def manifest_push_payload(manifest: Manifest, cluster: str) -> dict:
    """Project the manifest into one cluster's wire payload.

    The backend files each pushed payload under its single ``cluster`` string
    and the almanac reads the flat pre-v6 checkpoint shape, and a per-cluster
    projection *is* exactly the manifest a dedicated single-cluster install
    would have written — so the wire format stays schema v5 and neither needs
    any change. Envs that don't serve ``cluster`` are omitted; each
    checkpoint carries only this cluster's verification state (its verify
    error, or the shared download error when there is none).

    Cluster-specific variants shadow per id (#208): when a variant serving
    ``cluster`` records a checkpoint id, the universal env's row for that id
    is omitted from this cluster's payload. Smoke-test tests the variant's
    copy there (checkpoint-first selection), so keeping the universal row
    would advertise a permanently-unverified checkpoint. Shadowing reads the
    manifest's *records*, not the source declarations — this stays a pure
    function of the manifest, no disk access — so until the first
    checkpoint-first smoke-test/add run on that cluster writes the variant's
    record, an overridden id may transiently still appear under the universal
    env; that gap self-heals on the first run. Equal specificity (two
    universal envs, or two variants both serving the cluster) never shadows:
    resolution errors on that authoring mistake anyway, and a loud duplicate
    in the almanac beats silently dropping one.

    Key order matters only cosmetically, but it is kept identical to a v5
    manifest so dumps diff cleanly across the transition.
    """
    # Ids recorded by cluster-specific envs serving this cluster — these
    # shadow the universal envs' rows below.
    variant_ids: set[str] = set()
    for env in manifest.environments.values():
        if env.clusters is not None and cluster in env.clusters:
            variant_ids.update(env.checkpoints)

    environments: dict[str, dict] = {}
    for env_name, env in manifest.environments.items():
        if not env.serves(cluster):
            continue
        checkpoints = {}
        for ckpt_name, ckpt in env.checkpoints.items():
            if env.clusters is None and ckpt_name in variant_ids:
                continue
            v = ckpt.verification(cluster)
            checkpoints[ckpt_name] = {
                "fetched_at": ckpt.fetched_at,
                "verified_at": v.verified_at,
                "verified_device": v.verified_device,
                "last_error": v.last_error or ckpt.last_error,
                "weight_files": ckpt.weight_files,
                "weights_recorded_at": ckpt.weights_recorded_at,
            }
        environments[env_name] = {
            "built_at": env.built_at,
            "source_hash": env.source_hash,
            "source": env.source,
            "python_requires": env.python_requires,
            "dependencies": env.dependencies,
            "lock_hash": env.lock_hash,
            "checkpoints": checkpoints,
        }

    return {
        "schema_version": 5,  # the flat single-cluster shape the wire has always carried
        "cluster": cluster,
        "root": manifest.root,
        "maintainer": manifest.maintainer.to_dict(),
        "rootstock_version": manifest.rootstock_version,
        "python_version": manifest.python_version,
        "last_updated": manifest.last_updated,
        "environments": environments,
    }


def compute_source_hash(source_path: Path) -> str:
    """Compute SHA256 hash of environment source file."""
    content = source_path.read_bytes()
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def get_installed_versions(
    env_path: Path, only_packages: list[str] | None = None
) -> dict[str, str]:
    """
    Get installed package versions from a venv.

    Uses uv pip list --format=json to get package versions.

    Args:
        env_path: Path to the virtual environment directory
        only_packages: Optional list of package names to include (direct dependencies)

    Returns:
        Dict of package name -> version
    """
    python_path = env_path / "bin" / "python"
    if not python_path.exists():
        return {}

    # Normalize package names for matching if only_packages is provided
    def normalize(name: str) -> str:
        return name.lower().replace("-", "_").replace(".", "_")

    def base_name(spec: str) -> str:
        # Strip version specifiers and extras: "torch>=2.0" / "mace[dev]" -> package name
        for sep in ("==", ">", "<", "~", "["):
            spec = spec.split(sep)[0]
        return spec.strip()

    filter_set = {normalize(base_name(p)) for p in only_packages} if only_packages else None

    packages_data = []

    # Prefer uv pip list if uv is available
    if shutil.which("uv"):
        try:
            result = subprocess.run(
                ["uv", "pip", "list", "--python", str(python_path), "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                packages_data = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError):
            pass

    # Fallback to python -m pip list
    if not packages_data:
        try:
            result = subprocess.run(
                [str(python_path), "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                packages_data = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError):
            pass

    if not packages_data:
        return {}

    installed = {}
    for pkg in packages_data:
        name = pkg["name"]
        version = pkg["version"]
        if filter_set is None or normalize(name) in filter_set:
            installed[name] = version

    return installed


def detect_python_version(root: Path) -> str:
    """
    Detect Python version from installed interpreter.

    Args:
        root: Rootstock root directory

    Returns:
        Python version string (e.g., "3.10.19") or "unknown"
    """
    python_dir = root / ".python"
    if python_dir.exists():
        for item in python_dir.iterdir():
            if item.is_dir() and "cpython" in item.name:
                # Extract version from name like "cpython-3.10.19-linux-x86_64-gnu"
                parts = item.name.split("-")
                if len(parts) >= 2:
                    return parts[1]
    return "unknown"


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def built_at_estimate(env_path: Path) -> str:
    """Best-effort build time for an env the manifest doesn't know about.

    The env directory's mtime is set when the build wrote its last top-level
    entry, so it approximates the true build time. Only ``install`` knows the
    exact moment; everything else discovering an already-built env must not
    fabricate ``built_at=now``, which would poison the
    ``verified_at > built_at`` staleness comparison.
    """
    ts = env_path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _try_create_lock(lock_path: Path) -> bool:
    """One atomic O_EXCL attempt; the lock file records who holds it."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump(
            {"pid": os.getpid(), "host": socket.gethostname(), "created": now_iso()},
            f,
        )
    return True


def _describe_lock(lock_path: Path) -> str:
    try:
        holder = json.loads(lock_path.read_text())
        return (
            f"held by pid {holder.get('pid')} on {holder.get('host')} since {holder.get('created')}"
        )
    except (OSError, json.JSONDecodeError):
        return "holder unknown (lock file unreadable)"


@contextmanager
def manifest_lock(
    root: Path,
    timeout: float = LOCK_TIMEOUT,
    stale_after: float = LOCK_STALE_AFTER,
):
    """Hold ``{root}/manifest.json.lock`` for a read-modify-write cycle.

    Every writer must load the manifest fresh *inside* this lock, mutate, and
    save before releasing — holding a manifest object across the lock
    boundary reintroduces the lost-update race this exists to prevent
    (nightly smoke-test vs. a co-maintainer's add).

    Deliberately NOT fcntl/flock: the manifest lives on the install root,
    which can be a filesystem without working POSIX locks (Perlmutter CFS is
    why the cache_root split exists at all). An O_EXCL create is atomic on
    the filesystems we care about.

    A lock older than ``stale_after`` (far beyond any legitimate hold) is
    presumed abandoned by a killed process and broken: unlinked and
    re-contested. The mtime comes from the fileserver's clock, so
    ``stale_after`` is generous rather than tight.

    Raises:
        ManifestLockTimeout: not acquired within ``timeout`` seconds.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "manifest.json.lock"

    deadline = time.monotonic() + timeout
    while not _try_create_lock(lock_path):
        try:
            seen = lock_path.stat()
        except FileNotFoundError:
            continue  # released between attempts — retry immediately

        if time.time() - seen.st_mtime > stale_after:
            # Presumed abandoned. Re-stat before unlinking so we only break
            # the lock we judged stale, not one freshly taken since (the
            # unlink itself can still race another breaker — the loser of
            # the subsequent O_EXCL just keeps waiting).
            try:
                if lock_path.stat().st_mtime == seen.st_mtime:
                    lock_path.unlink()
            except FileNotFoundError:
                pass
            continue

        if time.monotonic() >= deadline:
            raise ManifestLockTimeout(
                f"could not lock {root / 'manifest.json'} after {timeout:.0f}s; "
                f"lock file {lock_path} is {_describe_lock(lock_path)}. "
                f"If that process is dead, remove the lock file and retry."
            )
        time.sleep(_LOCK_POLL_INTERVAL)

    try:
        yield
    finally:
        # If we exceeded stale_after ourselves, another process may have
        # broken our lock and taken its own; this unlink would then release
        # theirs early. Tolerated: stale_after is far above any real hold.
        lock_path.unlink(missing_ok=True)


def load_manifest(root: Path) -> Manifest | None:
    """
    Load manifest from root directory.

    Args:
        root: Rootstock root directory

    Returns:
        Manifest object, or None when no manifest.json exists.

    Raises:
        ManifestError: the file exists but is corrupt or unmigratable —
            never silently treated as missing.
    """
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            data = json.load(f)
        data, notes = migrate_manifest_data(data)
        if notes:
            for note in notes:
                print(f"Note: {note}", file=sys.stderr)
            print(
                "Note: the migrated manifest is written back on the next "
                "state-changing command (install/add/smoke-test).",
                file=sys.stderr,
            )
        return Manifest.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        # A manifest that exists but can't be parsed must NOT read as "no
        # manifest": callers would create a fresh one and the next save would
        # silently erase all fetch/verify history.
        raise ManifestError(
            f"manifest at {manifest_path} is corrupted "
            f"({type(exc).__name__}: {exc}); refusing to treat it as missing. "
            f"Fix the file, or move it aside to start fresh deliberately."
        ) from exc


def save_manifest(manifest: Manifest, root: Path) -> None:
    """
    Save manifest to root directory atomically.

    Uses atomic write (temp file + rename) to prevent corruption.

    Args:
        manifest: Manifest to save
        root: Rootstock root directory
    """
    manifest.last_updated = now_iso()
    manifest_path = root / "manifest.json"

    # Ensure root directory exists
    root.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename. mkstemp always creates
    # the file with 0600, which clamps the POSIX ACL mask to --- on shared
    # installs that rely on inherited default ACLs (e.g., NERSC project dirs).
    # Reset the mode to honor the process umask so group/other access can
    # come through the parent's default ACL.
    fd, temp_path = tempfile.mkstemp(dir=root, suffix=".json")
    try:
        with open(fd, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        umask_value = os.umask(0)
        os.umask(umask_value)
        os.chmod(temp_path, 0o666 & ~umask_value)
        Path(temp_path).rename(manifest_path)
    except Exception:
        # Clean up temp file on failure
        try:
            Path(temp_path).unlink()
        except OSError:
            pass
        raise


def create_manifest(
    root: Path,
    clusters: list[str],
    config: UserConfig,
) -> Manifest:
    """
    Create a new manifest for a rootstock installation.

    Args:
        root: Rootstock root directory
        clusters: Every cluster this install serves (e.g. ["delta"], or
            ["sophia", "polaris"] for a shared root); first entry is the
            install's "home" cluster
        config: User config with maintainer info

    Returns:
        New Manifest object (not yet saved)
    """
    from . import __version__

    return Manifest(
        schema_version=SCHEMA_VERSION,
        clusters=list(clusters),
        root=str(root),
        maintainer=Maintainer(
            name=config.name or "Unknown",
            email=config.email or "unknown@example.com",
        ),
        rootstock_version=__version__,
        python_version=detect_python_version(root),
        last_updated=now_iso(),
        environments={},
    )
