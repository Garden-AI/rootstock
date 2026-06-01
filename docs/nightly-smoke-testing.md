# Nightly Automated Smoke-Testing

This guide is for **cluster maintainers**. It covers the one-time setup needed
to bring a cluster into the nightly smoke-test rotation.

## Overview

Every night, a GitHub Actions workflow dispatches `rootstock smoke-test` to each
participating cluster. The path is:

```
GitHub Actions (cron, 07:00 UTC)
  └─ dispatches via groundhog-hpc, authenticating as a Globus confidential client
       └─ per-cluster Globus Compute endpoint (persistent login-node daemon)
            └─ submits a SLURM GPU job
                 └─ rootstock smoke-test --device cuda --json
                      └─ updates manifest.json and pushes it to the backend
```

The scheduling, dispatch, and authentication all live **off-cluster**. The only
on-cluster component is the Globus Compute endpoint, and the only thing this
guide sets up is that endpoint plus the machinery to keep it alive.

Do this once per cluster. The maintainer who owns the cluster's rootstock
install should also own the endpoint, so `rootstock smoke-test` runs as an
identity with `is_maintainer = true` and the manifest push works.

## Prerequisites

- A maintainer account on the cluster that is a member of a GPU allocation.
- Rootstock installed and configured on that account: `~/.config/rootstock/config.toml`
  with a valid API key and `is_maintainer = true`.
- Ability to install user-level Python packages (`pipx` or `pip install --user`).
- A way to keep a login-node process alive across crashes and reboots — a
  `cron` job (simplest) or a `systemd` user service with lingering. **Verify
  which your cluster allows early**; some disable both (see
  [Per-cluster notes](#per-cluster-notes)). The keep-alive mechanism shapes the
  rest of the setup.

## Step 1: Retrieve the Globus confidential-client credentials

Nightly CI authenticates as a Globus **confidential client** (a service
account), so it runs unattended with no browser flow. There is one client for
the whole project, shared across all clusters.

- **Client ID** (not secret, safe to commit):
  `84102a92-84b8-40ec-9520-4ece991654f1`
- **Client secret**: stored in AWS Secrets Manager. To mint a fresh one, open
  the `rootstock-nightly-runner` project at <https://app.globus.org/settings/developers>
  and click **Add Client Secret**; a client can hold several at once. AWS holds
  the canonical copy — reuse it rather than generating a new secret per cluster.

Retrieve the secret:

```bash
aws secretsmanager get-secret-value \
  --secret-id rootstock/nightly-smoke-test-client \
  --query SecretString --output text
```

The same `(client_id, client_secret)` pair is stored as the GitHub repo secrets
`GLOBUS_COMPUTE_CLIENT_ID` / `GLOBUS_COMPUTE_CLIENT_SECRET` that the Actions
workflow consumes. The credential is long-lived and does not need rotation —
only the access tokens minted from it expire, and the SDK refreshes those
transparently.

## Step 2: Install `globus-compute-endpoint` on the login node

```bash
uv tool install globus-compute-endpoint --python 3.13 # gce has issues with 3.14 
which globus-compute-endpoint        # note this path 
```

The install directory (usually `~/.local/bin`) matters later: the endpoint
re-execs this binary to spawn workers, so it must be reachable from the
service environment.

The binary ships a `gce` alias for `globus-compute-endpoint`; this guide spells
out the full name, but `gce start rootstock-nightly` and friends are equivalent.

## Step 3: Create and configure the endpoint

Name the endpoint `rootstock-nightly` to keep the systemd unit and these
instructions identical across clusters:

```bash
#TODO export env vars before this step!!
globus-compute-endpoint configure rootstock-nightly
```

This creates config files under `~/.globus_compute/rootstock-nightly/`. Copy the
template `scripts/nightly-endpoint-config.yaml.j2` (in the rootstock repo) to `~/.globus_compute/rootstock-nightly/user_config_template.yaml.j2` as a starting point, 
but config may need to change according to scheduler differences etc. The actual values for 
the fields templated here will be set by the groundhog script. 

There is no separate allowlist to configure. A Globus Compute endpoint is owned
by whichever identity starts it, and submission requires authenticating as that
same identity. Start the endpoint as the confidential client — credentials from
Step 1 in the environment — and it is owned by the service account; the nightly
dispatcher then submits simply by setting the *same*
`GLOBUS_COMPUTE_CLIENT_ID` / `GLOBUS_COMPUTE_CLIENT_SECRET`.

The credentials must therefore be set from the **very first start** — the
endpoint binds its owning identity then, and there is no way to change it later
short of recreating the endpoint.

Export the credentials, then start the endpoint once to record its UUID
(needed in Step 5):

```bash
export GLOBUS_COMPUTE_CLIENT_ID=84102a92-84b8-40ec-9520-4ece991654f1
export GLOBUS_COMPUTE_CLIENT_SECRET=<secret from Step 1>

globus-compute-endpoint start rootstock-nightly
globus-compute-endpoint list                  # copy the Endpoint ID
globus-compute-endpoint stop rootstock-nightly
```

Step 4 wires these same credentials into the keep-alive mechanism, so every
subsequent start uses them too.

## Step 4: Keep the endpoint alive

The endpoint is a persistent login-node daemon. It must come back after a crash,
a process sweep, or a login-node reboot. Two mechanisms work, depending on
cluster policy — verify which is available before you pick:

- **`cron`** (Option A): simplest, where plain `crontab` is allowed.
- **`systemd` user service with lingering** (Option B): the robust fallback,
  and the only option on clusters where `crontab` is blocked (e.g. Delta).

> SLURM `scrontab` is never the right tool here: scrontab jobs run on compute
> nodes, but the endpoint must live on the login node.

### Pick one login node first

Many clusters have several login nodes sharing `$HOME`. The endpoint must run on
**exactly one** of them — two endpoints sharing one UUID continuously evict each
other (see [Troubleshooting](#troubleshooting)). Whichever mechanism you choose,
pin it to a single login node. Run `hostname` on the node you pick and note it.

### Option A: cron

Where plain `crontab` is available, re-run `start` every 15 minutes. `start` is
a no-op when the endpoint is already running, so the job heals a crashed or
swept endpoint at the next tick. `crontab -e`, then:

```cron
# cron injects these assignment lines into every job's environment
GLOBUS_COMPUTE_CLIENT_ID=84102a92-84b8-40ec-9520-4ece991654f1
GLOBUS_COMPUTE_CLIENT_SECRET=<secret from Step 1>
PATH=/home/youruser/.local/bin:/usr/bin:/bin

*/15 * * * * globus-compute-endpoint start rootstock-nightly
```

The credentials **must** be in the job's environment — set them as crontab
assignment lines (above) so every run binds the service account, not your
identity. Use the binary's absolute path if you would rather not set `PATH`.

If the crontab spool is shared across login nodes, guard the command so only the
node you chose runs it:

```cron
*/15 * * * * [ "$(hostname)" = dt-login03 ] && globus-compute-endpoint start rootstock-nightly
```

That is all cron needs. Skip Option B.

### Option B: systemd user service

The robust, self-healing mechanism for clusters where `crontab` is blocked. A
systemd user service with lingering survives logout, crash, and reboot with no
polling.

#### 4b.1. Enable lingering on the chosen node

Lingering lets your user services keep running after you log out and start on
boot. Linger state is **per login node** — it lives in `/var/lib/systemd/linger/`,
which is local to each node even when `$HOME` is shared — so enable it only on
the node you chose:

```bash
loginctl enable-linger $USER
loginctl show-user $USER | grep Linger      # want: Linger=yes
```

If `enable-linger` is denied, escalate to the cluster's support desk — ask for
the sanctioned way to run a persistent Globus Compute endpoint.

#### 4b.2. Create the credentials env file

`~/.config/rootstock/rootstock-nightly.env` — **plain `KEY=value`, no quotes**.
This file is read by systemd's `EnvironmentFile=`, which is *not* a shell, so
shell quoting does not apply:

```
GLOBUS_COMPUTE_CLIENT_ID=84102a92-84b8-40ec-9520-4ece991654f1
GLOBUS_COMPUTE_CLIENT_SECRET=<secret from Step 1, unquoted>
```

```bash
chmod 600 ~/.config/rootstock/rootstock-nightly.env
```

#### 4b.3. Create the service unit

`~/.config/systemd/user/rootstock-endpoint.service`. Pin `ConditionHost` to the
login node you chose above:

```ini
[Unit]
Description=Rootstock nightly Globus Compute endpoint
# Pin to the single login node chosen above — full hostname, glob-friendly.
ConditionHost=dt-login03*

[Service]
Type=simple
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=%h/.config/rootstock/rootstock-nightly.env
ExecStartPre=-%h/.local/bin/globus-compute-endpoint stop rootstock-nightly
ExecStart=%h/.local/bin/globus-compute-endpoint start rootstock-nightly
ExecStop=%h/.local/bin/globus-compute-endpoint stop rootstock-nightly
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
```

#### 4b.4. Enable and verify

```bash
systemctl --user daemon-reload
systemctl --user enable --now rootstock-endpoint.service
systemctl --user status rootstock-endpoint.service     # expect: active (running)
```

From here on, let systemd own the endpoint — do **not** run
`globus-compute-endpoint start` by hand, or you will get a pid-file conflict.

> **Logs:** `journalctl --user` is often denied on HPC login nodes ("No journal
> files were opened due to insufficient permissions") — that is harmless and does
> not affect the service. The endpoint writes its own logs to
> `~/.globus_compute/rootstock-nightly/endpoint.log`; use that for
> debugging, and `systemctl --user status` for health.

## Step 5: Register the cluster in the dispatch script

Edit the PEP 723 metadata in `scripts/nightly_smoke_test.py`. Add or update the
`[tool.hog.<cluster>]` block:

```toml
# [tool.hog.delta]
# endpoint = "27687af7-a20e-477a-8d4e-b5a7a097f864"
# account = "bhhl-delta-gpu"
# scheduler_options = "#SBATCH --gpus-per-node=1"
# partition = "gpuA100x4"
```

Only `endpoint` uuid is mandatory; the rest mirror the endpoint config's SLURM
provider and may be omitted where the endpoint config already covers them.

## Step 6: Verify end-to-end

From your laptop, with the credentials exported. Dispatch goes through the
groundhog CLI (`hog run`), which authenticates as the confidential client — so
the same env vars must be set locally, or it will not authorize:

```bash
export GLOBUS_COMPUTE_CLIENT_ID=84102a92-84b8-40ec-9520-4ece991654f1
export GLOBUS_COMPUTE_CLIENT_SECRET=<secret from Step 1>

# cheap probe — runs hello_endpoint in a worker, no SLURM GPU job
hog run scripts/nightly_smoke_test.py hello -- --endpoint=<cluster-name>

# full run — submits the real GPU smoke-test
hog run scripts/nightly_smoke_test.py main -- <cluster-name>
```

Then confirm the manifest landed in the backend (`rootstock status`). Finally,
add the cluster to the matrix in `.github/workflows/nightly-smoke-test.yml`.

## Per-cluster notes

### NCSA Delta

- `crontab` (PAM-blocked) and `scrontab` (disabled) are both unavailable, so
  Option A is out; `systemd` user lingering works — use Step 4 Option B.
- CUDA 12 is the default toolkit; use `cu12` wheels.
- GPU charge code: `bhhl-delta-gpu` (the official Delta deployment account).
  Non-interactive GPU partition: `gpuA100x4`.
- Endpoint UUID: `27687af7-a20e-477a-8d4e-b5a7a097f864`.
- Delta has multiple login nodes sharing `$HOME`. The endpoint is pinned to
  **`dt-login03`** (`ConditionHost=dt-login03*`), and lingering is enabled only
  on that node.
- See `notes/Cluster Delta.md` in the umbrella workspace for filesystem,
  permissions, and allocation gotchas.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `exit code 83`, `FileNotFoundError ... /usr/sbin/globus-compute-endpoint` | Endpoint can't find its own binary to spawn a worker; service PATH lacks the install dir. | Add `Environment=PATH=` with `~/.local/bin` (Step 4b.3), `daemon-reload`, restart. |
| Dispatch auth succeeds but submission is rejected | Confidential client not authorized on the endpoint. | Authorize the client identity (Step 3). |
| Endpoint shows `Disconnected` in `globus-compute-endpoint list` | Crashed daemon left a stale pid file. | `ExecStartPre=-... stop` handles it; otherwise `stop` then `start` manually. |
| Service restart-loops every ~2 min; `endpoint.log` shows clean `Shutdown complete` every ~90 s; `systemctl --user show` reports `Result=timeout`, `MainPID=0` | Unit uses `Type=forking`, but `globus-compute-endpoint` runs in the foreground and never forks; systemd times out the start at `TimeoutStartSec` (90 s) and restart-loops. | Set `Type=simple` and remove `PIDFile=` (Step 4b.3); `daemon-reload`, restart. |
| Endpoint bounces; `endpoint.log` shows restarts with the same UUID but jumping PIDs | A second endpoint with the same UUID is running on another login node — they evict each other. | Pin the unit to one node with `ConditionHost=` (Step 4b.3); `stop` the strays on the other nodes. |
| Job runs but fails with `BadStateException ... Job is marked as MISSING since the workers failed to register`; block stdout shows the worker pool starting then "exiting normally" | Workers can't reach the interchange — `address_by_hostname` resolved the login hostname to a non-routable address (e.g. link-local IPv6). | Use `address_by_interface` with a verified `ifname` (Step 3); restart the endpoint. |
| `journalctl --user`: "insufficient permissions" | Per-user journal not readable on this node. | Harmless. Use `~/.globus_compute/<name>/endpoint.log` and `systemctl --user status`. |
| Endpoint runs as your user identity, not the service account | Service started without the credential env vars. | Confirm `EnvironmentFile=` path and that the env file is unquoted `KEY=value`. |
