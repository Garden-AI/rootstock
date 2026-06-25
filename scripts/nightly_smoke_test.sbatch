#!/usr/bin/env bash
# === EDIT THESE for your cluster (SLURM) ===================================
#SBATCH -J rootstock-nightly
#SBATCH -A CHANGEME-gpu-account
#SBATCH -p CHANGEME-gpu-partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o rootstock-nightly-%j.out
# ==========================================================================
#
# Self-scheduling nightly smoke-test for SLURM clusters.
#
# A single GPU batch job that (1) re-queues its own next run, then (2) runs
# `rootstock smoke-test`, which pushes the refreshed manifest to the backend.
# The push IS the result channel, so nothing off-cluster waits on this job.
# The only state that persists between runs is "a job is queued for next time,"
# which this script maintains by resubmitting itself.
#
# One-time kickoff (submit once from a login node; it self-perpetuates):
#   sbatch scripts/nightly_smoke_test.sbatch
#
# Stop the chain:
#   touch ~/.rootstock-nightly-stop     # next run won't reschedule
#   squeue --me -n rootstock-nightly    # find the PENDING next run
#   scancel <jobid>                     # cancel it
# (Remove the stop file to allow a fresh kickoff later.)
#
# Fast end-to-end test (reschedule in 3 min instead of next week):
#   RESCHEDULE_BEGIN=now+3minutes sbatch scripts/nightly_smoke_test.sbatch

set -euo pipefail

# --- Knobs (override via env at submit time) --------------------------------
# Where rootstock is installed on this cluster. smoke-test reads/writes
# manifest.json here. Required.
export ROOTSTOCK_ROOT="${ROOTSTOCK_ROOT:-/CHANGEME/path/to/rootstock}"

# Cadence. Default weekly: drift-free wall-clock time CADENCE_DAYS ahead at
# RUN_HOUR (not "now+24h", which would creep later every run). Set
# CADENCE_DAYS=1 for nightly. RESCHEDULE_BEGIN overrides the whole computation
# with any `sbatch --begin` spec (e.g. now+3minutes) for testing.
CADENCE_DAYS="${CADENCE_DAYS:-7}"
RUN_HOUR="${RUN_HOUR:-02:00}"
RESCHEDULE_BEGIN="${RESCHEDULE_BEGIN:-$(date -d "+${CADENCE_DAYS} days ${RUN_HOUR}" +%Y-%m-%dT%H:%M:%S)}"

# Drop this file to stop the chain (see header).
STOP_FILE="${STOP_FILE:-$HOME/.rootstock-nightly-stop}"

JOB_NAME="${SLURM_JOB_NAME:-rootstock-nightly}"

# --- Reschedule FIRST -------------------------------------------------------
# Queue the next run before doing any work, so a crash in the smoke-test can't
# silently break the chain. The next job sits PENDING until --begin; that
# pending job is the one you scancel to stop the chain.
#
# Resolve THIS script's original path: in a batch job $0 is the spool copy, so
# ask the controller for the Command it was launched with.
if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SELF_PATH="$(scontrol show job "$SLURM_JOB_ID" \
    | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)"
fi
SELF_PATH="${SELF_PATH:-$0}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "not running under SLURM; skipping reschedule (dry run)."
elif [[ -e "$STOP_FILE" ]]; then
  echo "stop file present ($STOP_FILE); chain ends here."
elif squeue --me --name="$JOB_NAME" --states=PENDING --noheader 2>/dev/null | grep -q .; then
  # Belt-and-suspenders against duplicate chains (e.g. a manual re-kickoff
  # while a run is already queued).
  echo "a '$JOB_NAME' job is already PENDING; not scheduling another."
else
  next_id="$(sbatch --begin="$RESCHEDULE_BEGIN" --parsable "$SELF_PATH")"
  echo "scheduled next run: job $next_id (begins $RESCHEDULE_BEGIN)"
fi

# --- Payload ----------------------------------------------------------------
# Keep the ACL mask rw- on anything we touch on shared filesystems; batch jobs
# don't reliably source ~/.bashrc where a umask would otherwise be set.
umask 0002

# HF token for any checkpoint whose setup() re-checks gating even with weights
# cached. smoke-test never downloads, so a missing token is a warning, not a
# hard error.
if [[ -f ~/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "WARNING: no HF token at ~/.cache/huggingface/token; gated checkpoints may fail." >&2
fi

# Ensure `rootstock` is on PATH here (activate the venv / module load as your
# cluster requires). Batch jobs start from a bare environment.
#   source ~/.venvs/rootstock/bin/activate

nvidia-smi | head -4 || true
echo "=== rootstock smoke-test (root=$ROOTSTOCK_ROOT) ==="

# Re-verify every fetched checkpoint on GPU and PUSH the manifest (no
# --no-push): this is an unattended substitute for CI and must leave the
# backend in the state CI would. --json + tee keeps a machine-readable copy
# alongside the streaming SLURM .out log.
rootstock smoke-test \
  --device cuda \
  --json \
  | tee "rootstock-nightly-${SLURM_JOB_ID:-manual}.json"
