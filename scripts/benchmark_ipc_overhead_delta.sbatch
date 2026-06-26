#!/usr/bin/env bash
# === Delta (NCSA) — IPC-overhead benchmark ================================
#SBATCH -J rootstock-bench
#SBATCH -A bhhl-delta-gpu
#SBATCH -p gpuA100x4
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
# Uses the installed `rootstock` CLI (no repo checkout needed). Relies on
# SLURM's default --export=ALL to carry your submit-shell PATH (so `rootstock`
# resolves) and HF_TOKEN into the job. Submit from a shell where `which
# rootstock` works.
#
# Submit:
#   sbatch scripts/benchmark_ipc_overhead_delta.sbatch
#
# Override knobs at submit time, e.g. a different model set or CPU+GPU:
#   CHECKPOINTS="mace-mp-0-medium uma-s-1p1" DEVICES="cuda cpu" \
#     sbatch scripts/benchmark_ipc_overhead_delta.sbatch

set -euo pipefail

export ROOTSTOCK_ROOT="${ROOTSTOCK_ROOT:-/work/hdd/data/rootstock}"

# Checkpoints to compare. Pick a spread of model sizes/families so we see how
# overhead varies by per-call compute cost. List what's installed first:
#   rootstock benchmark --root $ROOTSTOCK_ROOT --list
CHECKPOINTS="${CHECKPOINTS:-mace-mp-0-medium uma-s-1p1 7net-0}"
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
