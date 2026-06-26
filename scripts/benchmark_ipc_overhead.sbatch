#!/usr/bin/env bash
# === EDIT THESE for your cluster (SLURM) ===================================
#SBATCH -J rootstock-bench
#SBATCH -A CHANGEME-gpu-account
#SBATCH -p CHANGEME-gpu-partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o rootstock-bench-%j.out
# ==========================================================================
#
# On-demand GPU job that measures Rootstock's i-PI IPC overhead by comparing
# RootstockCalculator against the in-environment direct calculator, for a few
# checkpoints. See `rootstock benchmark --help` for the methodology and options.
#
# Unlike the nightly smoke-test this does NOT reschedule itself — it's a
# one-shot benchmark you run when you want numbers.
#
# Submit:
#   sbatch scripts/benchmark_ipc_overhead.sbatch
#
# Override knobs at submit time, e.g. a different model set or CPU+GPU:
#   CHECKPOINTS="mace-mp-0-medium uma-s-1p1" DEVICES="cuda cpu" \
#     sbatch scripts/benchmark_ipc_overhead.sbatch

set -euo pipefail

# --- Knobs (override via env at submit time) --------------------------------
# Where rootstock is installed on this cluster. Required.
export ROOTSTOCK_ROOT="${ROOTSTOCK_ROOT:-/CHANGEME/path/to/rootstock}"

# Checkpoints to compare. Pick a spread of model sizes/families so we see how
# overhead varies by per-call compute cost. List what's installed first:
#   rootstock benchmark --root $ROOTSTOCK_ROOT --list
CHECKPOINTS="${CHECKPOINTS:-mace-mp-0-medium uma-s-1p1 sevennet-0}"
DEVICES="${DEVICES:-cuda}"
SYSTEM="${SYSTEM:-Cu:256}"
CALLS="${CALLS:-100}"
WARMUP="${WARMUP:-10}"

umask 0002

# HF token for any checkpoint whose setup() re-checks gating even when cached.
if [[ -f ~/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Ensure `rootstock` is on PATH here (activate the venv / module load as your
# cluster requires). Batch jobs start from a bare environment.
#   source ~/.venvs/rootstock/bin/activate

nvidia-smi | head -4 || true
echo "=== IPC-overhead benchmark (root=$ROOTSTOCK_ROOT) ==="

OUT="rootstock-bench-${SLURM_JOB_ID:-manual}.json"

rootstock benchmark \
  --root "$ROOTSTOCK_ROOT" \
  --checkpoints $CHECKPOINTS \
  --devices $DEVICES \
  --system "$SYSTEM" \
  --calls "$CALLS" \
  --warmup "$WARMUP" \
  --out "$OUT"

echo "=== wrote $OUT ==="
