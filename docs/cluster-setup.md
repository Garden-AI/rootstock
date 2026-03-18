# Setting Up a New Cluster

This section is for people setting up Rootstock on a new cluster. All commands below are run **on the cluster itself** (SSH in first). You'll need write access to a shared filesystem location visible to your users.

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

Choose a location on an appropriate shared filesystem where users can read but only maintainers can write. Then run:

```bash
rootstock init
```

This will interactively prompt you for:

| Setting | Description |
|---------|-------------|
| **root** | The shared directory path (e.g., `/scratch/shared/rootstock`) |
| **api_key / api_secret** | Optional credentials for pushing the cluster manifest to the Rootstock dashboard |
| **maintainer name / email** | Identifies the maintainer for this installation |

!!! tip "Dashboard Integration"
    Contact a Rootstock maintainer if you want your cluster to appear on the [dashboard](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/). The API credentials are [Modal Proxy Auth Tokens](https://modal.com/docs/guide/webhook-proxy-auth).

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

Each `rootstock install` command:

1. Creates an isolated virtual environment under `{root}/envs/`
2. Installs the MLIP's dependencies
3. Optionally pre-downloads model weights (via `--models`)

This can take several minutes per environment depending on the MLIP.

!!! note "Finding Environment Files"
    See the [dashboard](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/) for environment files that are known to work — you can use these as a starting point for your cluster.

## Step 4: Register with the Dashboard (Optional)

If you configured API credentials during `rootstock init`, the manifest is pushed automatically when you install or update environments. If the push failed (e.g., due to network issues), you can retry:

```bash
rootstock manifest push
```

## Verifying the Installation

After setup, verify that everything works:

```bash
# Check status
rootstock status

# List all environments
rootstock list

# Test a specific model (if you have GPU access)
rootstock test --model mace --checkpoint medium
```

## Directory Structure

After setup, the rootstock root directory will look like this:

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

Check your API credentials and network connectivity:

```bash
# Verify credentials are configured
rootstock config show

# Retry push
rootstock manifest push --verbose
```
