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
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import UserConfig

SCHEMA_VERSION = 3


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


# One entry per historical schema version, upgrading one step. A schema bump
# without a migration here strands every deployed manifest of that vintage —
# add the migration in the same change as the bump.
MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


def migrate_manifest_data(data: dict) -> tuple[dict, list[str]]:
    """Upgrade a raw manifest dict to SCHEMA_VERSION, one step at a time.

    Returns the upgraded dict and human-readable notes about lossy steps
    (empty when the manifest was already current). Raises RuntimeError for
    manifests from a *newer* rootstock or with no migration path.
    """
    version = data.get("schema_version")
    if isinstance(version, str) and version.isdigit():
        version = int(version)  # v1 wrote schema_version as a string

    if not isinstance(version, int):
        raise RuntimeError(
            f"manifest has invalid schema_version={data.get('schema_version')!r}; "
            f"expected an integer <= {SCHEMA_VERSION}."
        )

    if version > SCHEMA_VERSION:
        raise RuntimeError(
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
            raise RuntimeError(
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
        return {"name": self.name, "email": self.email}

    @classmethod
    def from_dict(cls, data: dict) -> Maintainer:
        return cls(name=data["name"], email=data["email"])


@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint registered with an environment."""

    fetched_at: str | None = None       # ISO 8601, set when download succeeds
    verified_at: str | None = None      # ISO 8601, set when smoke test passes
    verified_device: str | None = None  # "cuda", "cpu", etc.
    last_error: str | None = None       # most recent error from add or smoke-test

    def to_dict(self) -> dict:
        return {
            "fetched_at": self.fetched_at,
            "verified_at": self.verified_at,
            "verified_device": self.verified_device,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointInfo:
        return cls(
            fetched_at=data.get("fetched_at"),
            verified_at=data.get("verified_at"),
            verified_device=data.get("verified_device"),
            last_error=data.get("last_error"),
        )


@dataclass
class EnvironmentInfo:
    """Metadata for a single built environment."""

    status: str  # "ready", "building", "error"
    built_at: str  # ISO 8601 timestamp
    source_hash: str  # "sha256:abc123..."
    source: str  # Full source code of the environment file
    python_requires: str  # ">=3.11"
    dependencies: dict[str, str]  # {"mace-torch": "0.3.6"}
    checkpoints: dict[str, CheckpointInfo] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "built_at": self.built_at,
            "source_hash": self.source_hash,
            "source": self.source,
            "python_requires": self.python_requires,
            "dependencies": self.dependencies,
            "checkpoints": {name: ckpt.to_dict() for name, ckpt in self.checkpoints.items()},
        }
        if self.error_message:
            d["error_message"] = self.error_message
        return d

    @classmethod
    def from_dict(cls, data: dict) -> EnvironmentInfo:
        checkpoints_data = data.get("checkpoints", {})
        checkpoints = {
            name: CheckpointInfo.from_dict(ckpt_data)
            for name, ckpt_data in checkpoints_data.items()
        }
        return cls(
            status=data["status"],
            built_at=data["built_at"],
            source_hash=data["source_hash"],
            source=data.get("source", ""),
            python_requires=data["python_requires"],
            dependencies=data["dependencies"],
            checkpoints=checkpoints,
            error_message=data.get("error_message"),
        )


def is_verified(env: EnvironmentInfo, ckpt: CheckpointInfo) -> bool:
    """Return True if checkpoint was verified after the env was last built."""
    if ckpt.verified_at is None:
        return False
    return ckpt.verified_at > env.built_at  # ISO 8601 sorts lexically


@dataclass
class Manifest:
    """Root manifest for a rootstock installation."""

    schema_version: int
    cluster: str
    root: str
    maintainer: Maintainer
    rootstock_version: str
    python_version: str
    last_updated: str  # ISO 8601 timestamp
    environments: dict[str, EnvironmentInfo] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "cluster": self.cluster,
            "root": self.root,
            "maintainer": self.maintainer.to_dict(),
            "rootstock_version": self.rootstock_version,
            "python_version": self.python_version,
            "last_updated": self.last_updated,
            "environments": {name: env.to_dict() for name, env in self.environments.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        data, _ = migrate_manifest_data(data)

        environments = {}
        for name, env_data in data.get("environments", {}).items():
            environments[name] = EnvironmentInfo.from_dict(env_data)

        return cls(
            schema_version=data["schema_version"],
            cluster=data["cluster"],
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
        if not self.cluster:
            return False, "Missing cluster"
        if not self.root:
            return False, "Missing root"
        if not self.maintainer.name:
            return False, "Missing maintainer name"
        if not self.maintainer.email:
            return False, "Missing maintainer email"

        # Validate environment status values
        valid_statuses = {"ready", "building", "error"}
        for env_name, env_info in self.environments.items():
            if env_info.status not in valid_statuses:
                return False, f"Invalid status '{env_info.status}' for {env_name}"

        return True, "OK"


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

    filter_set = (
        {normalize(p.split("==")[0].split(">")[0].split("<")[0].split("~")[0].split("[")[0].strip()) for p in only_packages}
        if only_packages
        else None
    )

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
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
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
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
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


def load_manifest(root: Path) -> Manifest | None:
    """
    Load manifest from root directory.

    Args:
        root: Rootstock root directory

    Returns:
        Manifest object, or None if not found
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
    except (json.JSONDecodeError, KeyError):
        # Invalid manifest - could log warning here
        return None


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
    cluster: str,
    config: UserConfig,
) -> Manifest:
    """
    Create a new manifest for a rootstock installation.

    Args:
        root: Rootstock root directory
        cluster: Cluster name (e.g., "della", "modal")
        config: User config with maintainer info

    Returns:
        New Manifest object (not yet saved)
    """
    from . import __version__

    return Manifest(
        schema_version=SCHEMA_VERSION,
        cluster=cluster,
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
