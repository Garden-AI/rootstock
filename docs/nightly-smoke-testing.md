# Automated Testing

This guide is for **cluster maintainers**. It covers bringing a cluster into the
automated smoke-test rotation.

## Overview

`rootstock smoke-test` re-verifies every fetched checkpoint (one GPU forward
pass each) and pushes the refreshed `manifest.json` to the backend. Running it
on a schedule keeps each cluster's manifest honest about checkpoint health.

User-supplied weights (`<family>:custom` checkpoints) are never smoke-tested
— there is nothing registered to verify. Every use is a fresh load through
the env's `setup_from_path` hook, so a bad file simply fails at server start.

The mechanism is a **self-scheduling batch job**: a single GPU job that first
re-queues its own next run, then runs the smoke-test. The only state that
persists between runs is "a job is queued for next time," which the job
maintains by resubmitting itself.

Two recipes with placeholders:

- `scripts/nightly_smoke_test.sbatch` — SLURM (e.g. Della, Delta)
- `scripts/nightly_smoke_test.pbs` — PBS Pro (e.g. ALCF Sophia)

> The command and these recipes keep the name "nightly" for continuity, but the
> default cadence is **weekly** (`CADENCE_DAYS=7`). 

## Prerequisites

- A maintainer account on the cluster with a GPU allocation.
- Rootstock installed and configured on that account
  (`~/.config/rootstock/config.toml` with a valid API key and
  `is_maintainer = true`) so the manifest push succeeds.
- `rootstock` reachable on `PATH` inside a batch job (activate the venv or
  `module load` in the marked spot in the script).

## Setup

1. **Copy the recipe** for your scheduler and edit the marked block at the top
   (account, partition/queue, walltime) and `ROOTSTOCK_ROOT`:

   ```bash
   # SLURM:
   $EDITOR scripts/nightly_smoke_test.sbatch   # set #SBATCH -A/-p, ROOTSTOCK_ROOT
   # PBS:
   $EDITOR scripts/nightly_smoke_test.pbs      # set #PBS -A/-q, filesystems, ROOTSTOCK_ROOT
   ```

2. **Kick off the chain once.** It self-perpetuates from there:

   ```bash
   sbatch scripts/nightly_smoke_test.sbatch          # SLURM
   qsub  scripts/nightly_smoke_test.pbs              # PBS (submit from the scripts/ dir)
   ```

3. **Verify a full cycle.** After the first run finishes, confirm:
   - a next run is queued — `squeue --me -n rootstock-nightly` (SLURM) /
     `qstat -u $USER` (PBS) shows a pending job (on PBS a job with a future
     start time sits in state `W`, waiting — not `Q`), and
   - the manifest landed — `rootstock status` shows fresh `verified_at` times.

## How the self-scheduling works

- **Resubmit-first.** The job queues its next run *before* the smoke-test, so a
  crash during verification can't break the chain. The only way the chain dies
  is a job that never *starts* (allocation lapsed / queue wedged).
- **Drift-free cadence.** The next run is a fixed wall-clock time
  (`CADENCE_DAYS` ahead at `RUN_HOUR`), not `now+24h`, so it doesn't creep later
  each run. SLURM uses `sbatch --begin=…`; PBS uses `qsub -a …`.
- **No duplicate chains.** Before resubmitting, the job checks for an
  already-pending instance by name and skips if found — so a stray manual
  kickoff won't fork the chain.

### Stopping the chain

```bash
touch ~/.rootstock-nightly-stop          # the next run will not reschedule
# then cancel the already-queued next run:
squeue --me -n rootstock-nightly ; scancel <jobid>    # SLURM
qstat -u $USER                  ; qdel    <jobid>      # PBS
```

Remove the stop file to allow a fresh kickoff later.

### Testing without waiting a week

```bash
RESCHEDULE_BEGIN=now+3minutes sbatch scripts/nightly_smoke_test.sbatch                    # SLURM
qsub -v RESCHEDULE_AT=$(date -d "+3 minutes" +%m%d%H%M) scripts/nightly_smoke_test.pbs    # PBS
```

Both override when generation 1 schedules generation 2, which is the handoff
worth testing. (On PBS, `qsub -a <time>` would only delay generation 1's own
start and exercises nothing about the chain.)

## Knobs

Override via env at submit time (SLURM: inline `VAR=val sbatch …`; PBS:
`qsub -v VAR=val …`):

| Variable | Default | Purpose |
|---|---|---|
| `ROOTSTOCK_ROOT` | (edit in script) | install root holding `manifest.json` |
| `CADENCE_DAYS` | `7` | days between runs (`1` = nightly) |
| `RUN_HOUR` | `02:00` | wall-clock time of day for scheduled runs |
| `STOP_FILE` | `~/.rootstock-nightly-stop` | presence ends the chain |
| `RESCHEDULE_BEGIN` / `RESCHEDULE_AT` | computed | override the next-run time directly (testing) |

## Per-cluster notes

- **Della** (SLURM, Princeton), **Delta** (SLURM, NCSA): use the `.sbatch`
  recipe. Delta blocks `crontab`/`scrontab`, which is exactly why the
  self-scheduling job (needing only normal job submission) is the right fit. Use
  GPU account `bhhl-delta-gpu`, partition `gpuA100x4` on Delta. See
  `notes/Cluster Delta.md` (umbrella workspace) for filesystem/ACL gotchas — the
  recipe already sets `umask 0002` for the shared-filesystem ACL mask.
- **Sophia** (PBS, ALCF): use the `.pbs` recipe. Keep
  `-l filesystems=home:eagle`; the recipe sets the ALCF outbound proxy
  defensively (smoke-test itself doesn't download).
- **Other clusters**: copy whichever recipe matches the scheduler. The only
  per-cluster edits are the directive block and `ROOTSTOCK_ROOT`.

> `date -d` in the recipes is GNU date (present on Linux HPC login/compute
> nodes). The recipes are not meant to run on macOS.

> **Smoke-test always uses default kwargs.** `smoke-test` calls each env's
> `setup()` with no extra kwargs, so a checkpoint that only works with
> non-default kwargs (e.g. a UMA checkpoint needing `task=omol`) will appear
> failing even though `add` succeeded. The remedy is to make the preferred
> kwargs the env's default in the env file.
