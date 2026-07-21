#!/usr/bin/env bash
# One-time setup for a rented AMD (MI300X) box with ROCm preinstalled.
# Developed on RunPod; any ROCm host works.
#
#   scp -r sample_model_configurations root@<box>:/workspace/code/
#   ssh root@<box> 'bash /workspace/code/sample_model_configurations/amd_workshop/bootstrap.sh'
#
# Then export HF_TOKEN (gated checkpoints) and start probing:
#   ssh root@<box>
#   export HF_TOKEN=hf_...
#   # Keep envs + wheel cache OFF the boot disk: a ROCm torch env is ~17 GB.
#   export WORKSHOP_ROOT=/workspace/rootstock-workshop UV_CACHE_DIR=/workspace/uv-cache
#   cd /workspace/code/sample_model_configurations/amd_workshop
#   python3 workshop.py probe ../amd_configs/mace.py --checkpoint mace-mp-0-medium
set -euo pipefail

echo "== GPU check =="
rocm-smi --showproductname || { echo "FATAL: rocm-smi failed - not a ROCm box?"; exit 1; }

echo "== Installing uv =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== torch-on-ROCm smoke test (throwaway env) =="
uv run --python 3.11 --index https://download.pytorch.org/whl/rocm6.4 --with torch -- python - <<'EOF'
import torch
print("torch", torch.__version__, "| ROCm/HIP:", torch.version.hip)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False - ROCm torch can't see the GPU"
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print("matmul OK, sum =", (x @ x).sum().item())
EOF

echo "== Bootstrap OK =="
