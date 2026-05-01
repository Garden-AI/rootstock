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

If you're adding a new cluster that needs the split, the maintainer creates the cache directory once with read access for everyone in the project (`chmod a+rx` on the directory tree, or appropriate group ACLs).

Then run the initialization command:

```bash
rootstock init
```

This will interactively prompt you for:

| Setting | Description |
|---------|-------------|
| **root** | The shared directory path (e.g., `/scratch/shared/rootstock`) |
| **api_key / api_secret** | Optional credentials for pushing the cluster manifest to the dashboard |
| **maintainer name / email** | Identifies the maintainer for this installation |

!!! tip "Dashboard Integration"
    Contact a Rootstock maintainer if you want your cluster to appear on the [Example Configs](clusters.md) page. The API credentials are [Modal Proxy Auth Tokens](https://modal.com/docs/guide/webhook-proxy-auth).

## Step 3: Install Environments

Still on the login node — `install` only builds the venv, no model weights yet:

```bash
# Install individual environments
rootstock install mace_env.py
rootstock install chgnet_env.py
rootstock install uma_env.py
rootstock install tensornet_env.py

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
rootstock add mace medium --no-verify
rootstock add uma uma-s-1p1 --no-verify --kwarg task=omat

# GPU node (no network): skip download (already fetched), verify on GPU
rootstock add mace medium
rootstock add uma uma-s-1p1 --kwarg task=omat
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
    See the [Example Configs](clusters.md) page for environment files that are known to work — you can use these as a starting point for your cluster.
    Some minor tweaks may be required depending on site specific requirements.

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
│   ├── mace_env.py
│   ├── chgnet_env.py
│   ├── uma_env.py
│   └── tensornet_env.py
├── envs/                   # Pre-built virtual environments
│   ├── mace_env/
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
rootstock install mace_env.py --force

# Re-verify checkpoints after the rebuild
rootstock add mace small
rootstock add mace medium
rootstock add mace large

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
