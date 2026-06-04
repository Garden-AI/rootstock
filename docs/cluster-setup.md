# Setting Up a New Cluster

This guide is for administrators setting up Rootstock on a new cluster. Run all commands below **on the cluster itself** after SSH access is established. Write access to a shared filesystem location visible to users is required.

## Prerequisites

- SSH access to the cluster
- Write access to a shared filesystem location
- Python 3.10 or later
- `uv` package manager (Rootstock uses it internally)

## Step 1: Install Rootstock

On a login node:

```bash
pip install rootstock
```

## Step 2: Initialize the Rootstock Directory

Choose a location on a shared filesystem where other users have access:

```bash
# Choose a shared directory path
# Example: /scratch/shared/rootstock
```

### Install root vs. cache root

On most clusters a single shared filesystem hosts both the rootstock install (code, venvs, manifest) and the model-weight cache. Some clusters require these to live on different filesystems — typically because the recommended persistent project filesystem doesn't support `flock`, which the HuggingFace cache requires. NERSC Perlmutter is one such case: code lives on CFS, model weights on PSCRATCH.

The cluster registry (`rootstock/clusters.py`) encodes both paths per cluster:

```python
"perlmutter": Cluster(
    root=Path("/global/cfs/cdirs/m4845/rootstock"),
    cache_root=Path("/pscratch/sd/w/wengler/rootstock-cache"),
),
```

When `cache_root` is omitted from the registry, both paths are the same. Users don't need to set environment variables — `RootstockCalculator(cluster="perlmutter", ...)` resolves both automatically.

### Permissions for shared installs

(We are trying to further automate this, but for the time being ...)

Shared installs on HPC clusters should be **world-readable**: anyone on the cluster (not just members of the maintainer's project group) should be able to use the environments and model weights. Nothing in a rootstock install is sensitive — it's all derived from public PyPI packages and public model checkpoints. Maintainer secrets (API tokens) live in the maintainer's `~/.config/rootstock/config.toml`, not in the shared root.

The setup needs to satisfy:

- Maintainers (and only maintainers) can write — `rootstock install` / `rootstock add` succeeds for them.
- Project group members inherit write via the maintainer's group, so a co-maintainer in the same project can take over without re-doing perms.
- All other cluster users get read + traverse.
- New files created by `uv pip install` (and by rootstock) inherit the project group, world-read, and group-write, so the above stays true going forward.

The recipe — run **once** as the maintainer, before `rootstock init`. Replace `<group>` with your project group (e.g., `m4845`).

```bash
# Install root: setgid + group-write for co-maintainers, world read+traverse
chmod 2775 /path/to/install/root
chgrp <group> /path/to/install/root
setfacl -m  g:<group>:rwx  /path/to/install/root
setfacl -dm g:<group>:rwx  /path/to/install/root   # default ACL — inherited by new files

# Cache root (only if separate filesystem). World-readable, only maintainer writes:
chmod 2755 /path/to/cache/root
chgrp <group> /path/to/cache/root
# Group inherits read+traverse from the mode bits; the setgid bit is just for
# group-ownership inheritance on new files. No named group ACL needed.
```

If the install or cache root **already has files in it** when you set this up (e.g., you're retrofitting a deployment that started out project-only), apply the same ACLs recursively so existing files become world-readable too:

```bash
setfacl -R -m  o::r-x  /path/to/install/root
setfacl -R -dm o::r-x  /path/to/install/root
setfacl -R -m  o::r-x  /path/to/cache/root
setfacl -R -dm o::r-x  /path/to/cache/root
```

Then **the maintainer's shell** needs `umask 002` so newly written files honor the inherited group-write bits. Add to `~/.bashrc` (or whatever rc the cluster sources for non-interactive shells):

```bash
umask 002
```

On clusters with split filesystems (cache root on a different mount than the install root), set:

```bash
export UV_LINK_MODE=copy
```

`uv` defaults to hardlinking from its cache into target venvs, which fails across filesystem boundaries and falls back to copy with a noisy warning. Setting `copy` mode silences the warning and is the correct mode for cross-filesystem builds anyway.

After `rootstock init` runs, verify ACLs landed correctly with `getfacl` on a freshly created file. Group should show `rwx` (effective `rw-` or `rwx`) and `mask::rw-` or stronger. If you see `mask::---` or `#effective:---`, the maintainer's umask was too restrictive when the file was created — rerun with `umask 002` and rewrite the file.

### Initial setup

Run the initialization command:

```bash
rootstock init
```

This will interactively prompt you for:

| Setting | Description |
|---------|-------------|
| **root** | The shared directory path, or a registered cluster name (`perlmutter`, `della`, etc.) |
| **api_key / api_secret** | Optional credentials for pushing the cluster manifest to the dashboard |
| **maintainer name / email** | Identifies the maintainer for this installation |

!!! tip "Dashboard Integration"
    If you provide API credentials, Rootstock pushes the cluster manifest to the dashboard automatically whenever you install or update environments. You can skip this and run a deployment that's never published.

## Step 3: Install Environments

Still on the login node — `install` only builds the venv, no model weights yet:

```bash
# Install individual environments
rootstock install mace.py
rootstock install uma.py
rootstock install tensornet.py

# Or point it at a directory with multiple environments
rootstock install ./environments/

# Verify everything is set up
rootstock status
```

Each `rootstock install` command:

1. Creates an isolated virtual environment under `{root}/envs/`
2. Installs MLIP dependencies

This process can take several minutes per environment, depending on the MLIP and network conditions.

## Step 4: Add Checkpoints

`rootstock add` is a separate, idempotent step that **downloads** weights and (where available) **verifies** them with a forward pass. Splitting download from verify lets you do the right thing on each kind of node:

```bash
# Login node (CPU, has network): download weights only
rootstock add mace-mp-0-medium --no-verify
rootstock add uma-s-1p1 --no-verify --kwarg task=omat

# GPU node (no network): skip download (already fetched), verify on GPU
rootstock add mace-mp-0-medium
rootstock add uma-s-1p1 --kwarg task=omat
```

If a node has both network access and a GPU, run without `--no-verify` to do everything in one shot.

`rootstock add` is idempotent — re-running it after a successful download will skip the download phase and just re-verify.

`rootstock smoke-test` re-verifies every fetched checkpoint and is suitable for nightly cron:

```bash
0 4 * * * rootstock smoke-test --json > /var/log/rootstock-smoke.log 2>&1
```

!!! note "Smoke-test always uses default kwargs"
    `smoke-test` calls each env's `setup()` with no extra kwargs. A checkpoint that only works with non-default kwargs (e.g., a UMA checkpoint that needs `task=omol`) will appear failing in nightly smoke-test even though `add` succeeded. The remedy is to make the preferred kwargs the env's default in the env file.

!!! note "Finding Environment Files"
    The live dashboard manifest exposes the environment source file for every deployed env; see the [Clusters](clusters.md) page. Copy a working source as a starting point for your cluster — some tweaks may be required for site-specific requirements.

## Step 5: Register with the Dashboard (Optional)

If you configured API credentials during `rootstock init`, the manifest is pushed automatically when you install or update environments.

### Managing the Manifest

The manifest tracks the state of your Rootstock installation and is used by the dashboard to display available environments. You can manage it with the following commands:

#### View Current Manifest

```bash
# Display the manifest in human-readable format
rootstock manifest show

# Output as JSON
rootstock manifest show --json
```

#### Push Manifest to Dashboard

If the automatic push failed (e.g., due to network issues), you can manually retry:

```bash
rootstock manifest push
```

#### Initialize a New Manifest

To create or reinitialize a manifest for a cluster:

```bash
# Create a new manifest
rootstock manifest init --cluster della

# Overwrite existing manifest
rootstock manifest init --cluster della --force

# Skip automatic push to backend
rootstock manifest init --cluster della --no-push
```

## Verifying the Installation

After setup, verify that everything works:

```bash
# Check status
rootstock status

# List all environments
rootstock list
```

## Directory Structure

After setup, the Rootstock root directory will look like this:

```
{root}/
├── .python/                # uv-managed Python interpreters
├── environments/           # Environment source files (*.py with PEP 723 metadata)
│   ├── mace.py
│   ├── uma.py
│   └── tensornet.py
├── envs/                   # Pre-built virtual environments
│   ├── mace/
│   │   ├── bin/python
│   │   ├── lib/python3.11/site-packages/
│   │   └── env_source.py
│   └── ...
├── home/                   # Redirected HOME for not-well-behaved libraries
│   ├── .cache/fairchem/
│   └── .matgl/
└── cache/                  # XDG_CACHE_HOME and HF_HOME for well-behaved libraries
    ├── mace/
    └── huggingface/
```

## Updating Environments

To update an environment with new dependencies:

```bash
# Rebuild the venv (drops verification timestamps for that env's checkpoints)
rootstock install mace.py --force

# Re-verify checkpoints after the rebuild, by canonical id
rootstock add mace-mp-0-small
rootstock add mace-mp-0-medium
rootstock add mace-mp-0-large

# Or re-verify every fetched checkpoint at once
rootstock smoke-test

# Push updated manifest
rootstock manifest push
```

Rebuilding an env invalidates prior verifications (the venv changed; weights in `cache/` are unaffected). `rootstock status` will show those checkpoints as **stale** until you re-run `add` or `smoke-test`.

## Troubleshooting

### Environment build fails

Check that you have:

- Sufficient disk space in `{root}/`
- Network access for downloading packages and model weights
- Correct Python version (3.10+)

### Users can't access environments

Verify permissions:

```bash
# Environments should be readable by all users
ls -la {root}/envs/

# Model weights in cache should also be readable
ls -la {root}/cache/
```

### Dashboard push fails

Check your API credentials and network connectivity, then retry the push:

```bash
rootstock manifest push
```
