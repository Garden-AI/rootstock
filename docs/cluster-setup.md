# Setting up a new cluster

This guide is for administrators setting up Rootstock on a new cluster. Run all commands below **on the cluster itself** after SSH access is established. Write access to a shared filesystem location visible to users is required.

## Prerequisites

- SSH access to the cluster
- Write access to a shared filesystem location
- Python 3.11 or later
- `uv` package manager (Rootstock uses it internally)

## Step 1: Install Rootstock

On a login node:

```bash
pip install rootstock
```

## Step 2: Initialize the Rootstock directory

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

Shared installs on HPC clusters should be **world-readable**: anyone on the cluster (not just members of the maintainer's project group) should be able to use the environments and model weights. Nothing in a rootstock install is sensitive — it's all derived from public PyPI packages and public model checkpoints. Maintainer secrets (API tokens) live in the maintainer's `~/.config/rootstock/config.toml`, not in the shared root.

The setup needs to satisfy:

- Maintainers (and only maintainers) can write — `rootstock install` / `rootstock add` succeeds for them.
- Project group members inherit write via the maintainer's group, so a co-maintainer in the same project can take over without re-doing perms.
- All other cluster users get read + traverse.
- New files created by `uv pip install` (and by rootstock) inherit the project group, world-read, and group-write, so the above stays true going forward.

`rootstock setup-perms` renders and applies this recipe for you. Run it **once** as the maintainer, before `rootstock init`. Pass a registered cluster name (it resolves both the install root and, where split, the cache root) and your project group:

```bash
rootstock setup-perms --cluster perlmutter --group m4845 --apply
```

Or point it at explicit paths instead of a cluster:

```bash
rootstock setup-perms /path/to/install/root \
  --cache-root /path/to/cache/root \
  --group <group> --apply
```

Omit `--apply` for a **dry run** (the default) that just prints the `chmod` / `chgrp` / `setfacl` commands — useful if you (or a cautious sysadmin) want to review them or paste them into a script before anything touches the filesystem. `--apply` runs them after a confirmation prompt, stopping at the first failure.

If the install or cache root **already has files in it** when you set this up (e.g., you're retrofitting a deployment that started out project-only), add `--retrofit` so the recipe also applies recursively and existing files become world-readable too:

```bash
rootstock setup-perms --cluster perlmutter --group m4845 --retrofit --apply
```

`rootstock install` re-checks these permissions up front on every run and prints a warning if the root doesn't look world-readable (wrong mode bits, missing setgid, missing default ACL, or a mask clamp from too-restrictive a umask). The check is read-only and never blocks the build; pass `--no-perm-check` to silence it.

`rootstock install` and `rootstock add` force `umask 002` for their own duration (uv subprocesses inherit it), so files they create are born world-readable and group-writable regardless of your personal umask. For anything you write into the shared root *by hand* (e.g. dropping an env file into `environments/`), your shell still needs `umask 002`. Add to `~/.bashrc` (or whatever rc the cluster sources for non-interactive shells):

```bash
umask 002
```

On clusters with split filesystems (cache root on a different mount than the install root), set:

```bash
export UV_LINK_MODE=copy
```

`uv` defaults to hardlinking from its cache into target venvs, which fails across filesystem boundaries and falls back to copy with a noisy warning. Setting `copy` mode silences the warning and is the correct mode for cross-filesystem builds anyway.

After `rootstock init` runs, verify ACLs landed correctly with `getfacl` on a freshly created file. Group should show `rwx` (effective `rw-` or `rwx`) and `mask::rw-` or stronger. If you see `mask::---` or `#effective:---`, the maintainer's umask was too restrictive when the file was created — rerun with `umask 002` and rewrite the file.

### Verifying world-readability before launch

Per-cluster step-by-step runbooks live in `scripts/runbooks/` — they sequence the checks below and record which findings are known-benign. The pieces:

**Quick check (first line).** `rootstock check-perms` runs the same read-only verification that `rootstock install` performs up front, as a standalone command — plus an ancestor walk, so a restricted project parent directory (which no `chmod` inside the install can fix) shows up too. It stats only the roots and their ancestors, so it is safe and fast on login nodes:

```bash
rootstock check-perms --cluster perlmutter --group m4845
```

Exit code 0 means the roots look right; 1 means issues were printed (pass `--json` for machine-readable output). It checks only the root directories, not the tree beneath them — for that, use the audit below.

**Static audit (full tree).** `scripts/check_world_readable.sh` checks the world-readable contract across the whole tree without needing rootstock in the caller's Python:

```bash
./scripts/check_world_readable.sh /global/cfs/cdirs/m4845/rootstock
```

It walks the ancestor directories (every one needs `o+x` — if the project parent on CFS lacks it, that's a facilities ticket, not a chmod), checks other-bits on every file and directory, resolves symlink targets, and scans for per-user ACL entries and mask clamps that `ls -l` won't show. It prints actionable per-path fixes and exits nonzero on any violation.

**Functional test (the ground truth).** Static checks can't prove the contract for the person who matters: have someone who does *not* own the install and is *not* in the project group load a model end-to-end through `RootstockCalculator`. The account matters, and there is no way to fake it: owner bits mask everything for whoever built the tree, group bits/ACLs mask everything for project-group members, and user namespaces (`unshare`) hide group membership without dropping it.

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

## Step 3: Install environments

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

## Step 4: Add checkpoints

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

`rootstock smoke-test` re-verifies every fetched checkpoint. To keep a cluster's
manifest current automatically, set it up on a schedule — see
[Nightly Automated Smoke-Testing](nightly-smoke-testing.md) for the recommended
approach (a self-scheduling GPU batch job, with ready-to-edit SLURM and PBS
recipes). It needs no cron and no off-cluster pieces, and is portable across sites.

!!! note "Smoke-test always uses default kwargs"
    `smoke-test` calls each env's `setup()` with no extra kwargs. A checkpoint that only works with non-default kwargs (e.g., a UMA checkpoint that needs `task=omol`) will appear failing in nightly smoke-test even though `add` succeeded. The remedy is to make the preferred kwargs the env's default in the env file.

!!! note "Finding Environment Files"
    The live dashboard manifest at [`garden-ai-prod--rootstock-admin-dashboard.modal.run`](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/) exposes the environment source file for every deployed env. Copy a working source as a starting point for your cluster — some tweaks may be required for site-specific requirements.

## Step 5: Register with the dashboard (optional)

If you configured API credentials during `rootstock init`, the manifest is pushed automatically when you install or update environments.

### Managing the manifest

The manifest tracks the state of your Rootstock installation and is used by the dashboard to display available environments. You can manage it with the following commands:

#### View current manifest

```bash
# Display the manifest in human-readable format
rootstock manifest show

# Output as JSON
rootstock manifest show --json
```

#### Push manifest to dashboard

If the automatic push failed (e.g., due to network issues), you can manually retry:

```bash
rootstock manifest push
```

#### Initialize a new manifest

To create or reinitialize a manifest for a cluster:

```bash
# Create a new manifest
rootstock manifest init --cluster della

# Overwrite existing manifest
rootstock manifest init --cluster della --force

# Skip automatic push to backend
rootstock manifest init --cluster della --no-push
```

## Verifying the installation

After setup, verify that everything works:

```bash
# Check status
rootstock status

# List all environments
rootstock list
```

## Directory structure

After setup, the Rootstock root directory will look like this:

```
{root}/
├── layout.json             # on-disk layout version (future clients check this)
├── .python/                # uv-managed Python interpreters
├── environments/           # Environment source files (*.py with PEP 723 metadata)
│   ├── mace.py
│   ├── mace.py.lock        # uv lockfile — rebuilds resolve from this
│   ├── uma.py
│   ├── uma.py.lock
│   └── tensornet.py
├── envs/                   # Pre-built virtual environments
│   ├── mace/
│   │   ├── bin/python
│   │   ├── lib/python3.11/site-packages/
│   │   ├── env_source.py
│   │   └── env_source.py.lock   # what this build was resolved from
│   └── ...
├── wheels/                 # Vendored rootstock wheels (rebuilds install from here,
│                           # so they don't depend on PyPI still serving the release)
├── home/                   # Redirected HOME for not-well-behaved libraries
│   ├── .cache/fairchem/
│   └── .matgl/
└── cache/                  # XDG_CACHE_HOME and HF_HOME for well-behaved libraries
    ├── mace/
    └── huggingface/
```

### Why the `home/` directory?

Some ML libraries (FAIRChem, MatGL) ignore `XDG_CACHE_HOME` and write to `~/.cache/` unconditionally. Rootstock redirects `HOME` during environment builds and worker runtime so model weights land in the shared install directory rather than in individual users' home directories.

## Updating environments

Rebuilds honor the env's lockfile by default: `rootstock install mace --force` reproduces the dependency stack that was already qualified on the cluster (this is the safe way to roll out an env-source fix). To deliberately move to newer packages, re-resolve with `--upgrade`:

```bash
# Rebuild the venv (drops verification timestamps for that env's checkpoints).
# Add --upgrade to re-resolve dependencies to the latest allowed versions.
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

Rebuilds are safe on a live shared install: the new env is built in `{root}/.build/` and swapped into `envs/` only when finished, so users can keep spawning workers from the old env for the whole build, and a **failed** rebuild leaves the old env untouched and working.

### Hotfixing `setup()` without a rebuild

`envs/<name>/env_source.py` is re-read at runtime on both sides of the socket: every worker spawn imports `setup()` from it, and every client resolution AST-parses its `CHECKPOINTS` table. Nothing caches it across runs — which means a maintainer can fix a bug in `setup()` (or adjust a `CHECKPOINTS` entry) by **editing that file in place on the shared filesystem, with no rebuild**. The next worker spawn picks it up. This is the one cheap lever into already-built envs, and it is a supported procedure:

1. **Edit a copy, then move it into place** (an in-progress edit with a syntax error breaks every new worker spawn on the cluster):
   ```bash
   cp {root}/envs/mace/env_source.py /tmp/env_source.py
   $EDITOR /tmp/env_source.py
   # umask 002 so the result stays world-readable
   mv /tmp/env_source.py {root}/envs/mace/env_source.py
   ```
2. **Apply the same edit to the registered source** `{root}/environments/mace.py` — that file is what rebuilds build from, so skipping this step means the next `install --force` silently reverts your hotfix.
3. **Verify and refresh the manifest**: `rootstock smoke-test --env mace` (or `rootstock add <id>`) exercises the fixed `setup()` and pushes an updated manifest. Until this runs, the manifest's `source_hash` for the env is stale — it still hashes the pre-hotfix source.
4. **Contribute the fix back** to the sample env file in the repo so the next cluster doesn't need the same hotfix.

**What a hotfix can and cannot reach.** In-place edits can change anything `setup()` does (model loading logic, upstream URLs/paths, `CHECKPOINTS` values, new optional kwargs) — but they cannot change the env's **dependencies** (the venv is already built; dependency changes need `install --force`) and cannot fix **worker/protocol code** (that's the rootstock pinned inside the venv; only a rebuild replaces it).

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
