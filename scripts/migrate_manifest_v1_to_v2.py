#!/usr/bin/env python3
"""
One-shot migration: rootstock manifest schema v1 -> v2.

# TODO(rootstock-v0.8.0+1): delete this script after known clusters are migrated.

v1 stored `checkpoints: list[str]`. v2 stores `checkpoints: dict[str, dict]`
where each value is a `CheckpointInfo` (fetched_at / verified_at /
verified_device / last_error). v2 also makes `schema_version` an int (was a
string in v1).

For each existing checkpoint name, we mint an empty CheckpointInfo. Nothing
is fetched and nothing is verified yet — operators run `rootstock add` /
`rootstock smoke-test` after migration to populate the timestamps.

Usage:
    python scripts/migrate_manifest_v1_to_v2.py <path-to-manifest.json>

Idempotent: re-running on an already-v2 manifest is a no-op.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def migrate(data: dict) -> dict:
    version = data.get("schema_version")
    # Accept either string "1" or int 1 for robustness.
    if version in (2, "2"):
        return data
    if version not in (1, "1"):
        raise SystemExit(
            f"Unexpected schema_version: {version!r}. Expected 1 or 2."
        )

    new_envs = {}
    for env_name, env in data.get("environments", {}).items():
        ckpt_list = env.get("checkpoints", []) or []
        new_ckpts = {
            name: {
                "fetched_at": None,
                "verified_at": None,
                "verified_device": None,
                "last_error": None,
            }
            for name in ckpt_list
        }
        new_env = dict(env)
        new_env["checkpoints"] = new_ckpts
        new_envs[env_name] = new_env

    out = dict(data)
    out["schema_version"] = 2
    out["environments"] = new_envs
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <path-to-manifest.json>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        return 1

    with path.open() as f:
        data = json.load(f)

    if data.get("schema_version") in (2, "2"):
        print(f"{path} is already schema_version=2. Nothing to do.")
        return 0

    backup = path.with_suffix(path.suffix + ".v1.bak")
    shutil.copy2(path, backup)
    print(f"Backed up original to {backup}")

    new_data = migrate(data)
    with path.open("w") as f:
        json.dump(new_data, f, indent=2)

    n_envs = len(new_data.get("environments", {}))
    n_ckpts = sum(len(e.get("checkpoints", {})) for e in new_data["environments"].values())
    print(f"Migrated {path}: {n_envs} envs, {n_ckpts} checkpoints (all unverified).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
