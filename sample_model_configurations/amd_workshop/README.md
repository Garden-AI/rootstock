# AMD workshop - MLIP configs on ROCm, targeting Frontier

The ROCm counterpart of `../modal_app.py`. Same philosophy - a workshop, not a
validator: the artifact we ship is the config in `../amd_configs/`.

**Why it exists.** Every cluster in the almanac is NVIDIA; ORNL's Frontier is
AMD MI250X, and every config in `nvidia_configs/` hardcodes a CUDA wheel index.
Modal has no AMD GPUs, so the Modal workshop cannot derive the ROCm equivalents.
This workshop runs the same loop on a rented MI300X instead (developed on
RunPod; any ROCm box works).

**The MI300X is a surrogate, not the target.** Frontier is MI250X (`gfx90a`);
the rentable AMD box is MI300X (`gfx942`). GPU code is compiled per-architecture,
so that gap is real - it is the entire reason `check_isa.py` exists.

## Layout

| | |
|---|---|
| `workshop.py` | build a venv per config from its PEP 723 block, run `../_agent/probe.py` in it |
| `check_isa.py` | **Frontier gate** - does the env actually contain `gfx90a` machine code? |
| `check_numerics.py` | **correctness gate** - do source-built GPU ops match CPU? |
| `bootstrap.sh` | one-time box setup (rocm-smi, uv, torch-on-ROCm smoke test) |

## Modal -> AMD box

| Modal | here |
|---|---|
| `modal.Image` + `uv_pip_install` per MLIP | one uv venv per config - the same path `rootstock install` uses on HPC |
| `modal run app::probe_x` | `ssh` + `workshop.py probe`; stdout streams natively |
| `modal.Volume` at `/cache` | a directory on the box's data volume, same HOME/HF_HOME redirection |
| `modal.Secret("huggingface")` | `export HF_TOKEN=...` |
| `gpu="A10G"` | the box *is* the GPU (192 GB - the A10G OOM failure modes vanish) |

`_agent/probe.py` is unchanged: PyTorch's ROCm build exposes AMD GPUs through
the `cuda` API, so `--device cuda` selects the AMD card, and every `setup()` in
`amd_configs/` is byte-identical to its `nvidia_configs/` twin. **The entire
port lives in the PEP 723 dependency block.**

## Quickstart

```bash
scp -r sample_model_configurations root@<box>:/workspace/code/
ssh root@<box>
bash /workspace/code/sample_model_configurations/amd_workshop/bootstrap.sh

export HF_TOKEN=hf_...                       # gated checkpoints (UMA)
export WORKSHOP_ROOT=/workspace/rootstock-workshop
export UV_CACHE_DIR=/workspace/uv-cache      # NOT the boot disk - see Storage
cd /workspace/code/sample_model_configurations/amd_workshop

python3 workshop.py probe ../amd_configs/mace.py --checkpoint mace-mp-0-medium
python3 check_isa.py mace --require gfx90a
```

Iteration loop (same as the Modal workshop):

1. Copy the closest `nvidia_configs/<name>.py` into `amd_configs/` and repoint
   torch at the ROCm index via `[tool.uv.sources]` / `[[tool.uv.index]]`
   (see `amd_configs/mace.py` - and note it must also declare
   `pytorch-triton-rocm`, which exists on no other index).
2. `workshop.py probe ../amd_configs/<name>.py --checkpoint <id>` - watch the
   `STAGE:` markers.
3. On failure, grep the error against `../_agent/failure_modes.md`; fix; re-run
   (`--fresh` after dependency changes).
4. On success: run `check_isa.py`, then commit the config plus any new
   failure-mode entry.

**Storage:** a ROCm torch env is **~17 GB** (the wheel alone is 4.2 GB - roughly
double CUDA's, since it fat-binaries kernels for every gfx target). Five envs
plus the shared wheel cache wants ~100 GB. Put `WORKSHOP_ROOT` and
`UV_CACHE_DIR` on a data volume; a full boot disk surfaces as a baffling
mid-install `No space left on device` on some unrelated package.

## The Frontier gap (read this)

Code built only for `gfx942` **imports fine, probes green, and then dies on
Frontier** with `HSA_STATUS_ERROR_INVALID_ISA`. **The probe structurally cannot
catch this** - it passes on the build box. So gate on it:

```bash
python3 check_isa.py <env> --require gfx90a
```

Verified here: official ROCm torch wheels are fat binaries and **do** carry
`gfx90a` (23/23 fat libraries). So pure-wheel models are Frontier-safe for free.

The gap only bites what you compile yourself - `torch-scatter`, `torch-sparse`,
`torch-cluster`, for which **no ROCm wheels exist anywhere** (PyG publishes CUDA
only). Build those with `PYTORCH_ROCM_ARCH="gfx90a;gfx942"` so one artifact
serves both machines. You do **not** need an MI250X to compile for one.

## A build that succeeds is not a build that is correct

`torch-sparse` compiles on ROCm, imports, runs - and returns **wrong numbers**.
Its `spmm` kernel is built around a 32-lane NVIDIA warp; AMD wavefronts are 64
lanes, so its shuffle-based row reduction drops contributions:

```
dense reference  : [[2, 2, 2], [7, 7, 7]]
torch_sparse GPU : [[2, 2, 2], [2, 2, 2]]     <-- no error raised
```

The probe passes. The physics is corrupt. So for any source-built extension:

```bash
python3 check_numerics.py <env>     # every source-built GPU op vs a CPU reference
```

Verified: `torch-scatter` and `torch-cluster` are numerically correct on ROCm;
`torch-sparse`'s `spmm` is not.

See `../_agent/failure_modes.md` for the full taxonomy.
