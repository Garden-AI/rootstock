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

Still on the login node:

```bash
# Install individual environments
rootstock install mace_env.py --models small,medium
rootstock install chgnet_env.py
rootstock install uma_env.py --models uma-s-1p1
rootstock install tensornet_env.py

# Or point it at a directory with multiple environments
rootstock install ./environments/

# Verify everything is set up
rootstock status
```

Each `rootstock install` command performs the following:

1. Creates an isolated virtual environment under `{root}/envs/`
2. Installs MLIP dependencies
3. Pre-downloads model weights (when `--models` is specified)

This process can take several minutes per environment, depending on the MLIP and network conditions.

!!! note "Finding Environment Files"
    See the [Example Configs](clusters.md) page for environment files that are known to work — you can use these as a starting point for your cluster.
    Some minor tweaks may be required depending on site specific requirements.

## Step 4: Register with the Dashboard (Optional)

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

To update an environment with new dependencies or model weights:

```bash
# Rebuild with new models
rootstock install mace_env.py --models small,medium,large --force

# Push updated manifest
rootstock manifest push
```

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
