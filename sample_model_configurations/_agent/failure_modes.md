# MLIP env failure modes

A taxonomy of failures we've hit (or expect) when porting an MLIP into rootstock.
Each entry: **signature** → **likely cause** → **fix**. Add new entries as we
discover them. Format is deliberately terse for fast pattern-matching by an
agent.

---

## Dependency resolution

**Signature:** `uv pip install` errors mentioning torch/cuda wheel mismatch,
e.g. `No solution found ... torch-scatter ... cu121` while torch resolved to
`>=2.5`.
**Cause:** PyG/torch-scatter `find-links` index pinned to a specific
`torch-X.Y+cuXXX` URL doesn't satisfy whatever torch version actually got
resolved.
**Fix:** Either (a) pin torch in `dependencies` to match the find-links wheel
(`torch>=2.4,<2.5`), or (b) bump the find-links URL to match the torch
version uv chose. Read `uv pip install --verbose` output to see what torch
got picked.

**Signature:** Build hangs for >5 min during `uv pip install` with no
output.
**Cause:** Resolving a large dep graph (fairchem-core pulls ~80 deps).
**Fix:** Pass `--verbose` to `rootstock install` so uv prints per-package
download progress. Not a real failure; just noisy.

**Signature:** `No solution found ... Because the current Python version
(3.10.x) does not satisfy Python>=3.11,<3.14 and fairchem-core==2.20.0
depends on Python>=3.11`.
**Cause:** fairchem-core 2.20+ dropped Python 3.10; a config declaring an
older `requires-python` floor (or an old probe image default) no longer
resolves. The probe image default and all shipped configs declare `>=3.11`
as of the rootstock 3.11 bump, so this only bites stale local copies.
**Fix:** Set `python_version="3.11"` on the `@probe_image(...)` and bump the
config's PEP 723 `requires-python = ">=3.11"`.

**Signature:** `Failed to download and build '<name> @ git+...'` →
`Package metadata name '<other-name>' does not match given name '<name>'`.
**Cause:** A git/VCS requirement's PEP 508 name must match the distribution
name the repo *declares*, not the repo slug. `graph_electrostatics` (repo) is
published as `graph-longrange`.
**Fix:** Use the declared distribution name in the requirement:
`graph-longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git`.
The imported module (`graph_longrange`) is a third, separate name — don't
assume any of the three match.

---

## HuggingFace auth & cache

**Signature:** `OSError: ... 401 Client Error ...` or `gated repo` on first
model download.
**Cause:** `HF_TOKEN` not set in container, or token lacks gated-repo
permission, or model license not accepted.
**Fix:** Add `secrets=[modal.Secret.from_name("huggingface")]` to the
modal function. Verify token has gated access for the specific model org.

**Signature:** Model re-downloads on every run despite previous build
succeeding.
**Cause:** Worker process spawned without `HOME` / `HF_HOME` /
`HF_HUB_CACHE` pointing into `{root}/cache/huggingface/`. The build uses
`get_model_cache_env(root)`; the runtime path must too.
**Fix:** Verify `EnvironmentManager.get_environment_variables()` is called and
its result is passed to the worker subprocess. See
`rootstock/environment.py:get_model_cache_env`.

**Signature:** `xet` permission errors on cache writes.
**Cause:** Same as above — env vars not pointing at writable shared cache.

---

## FAIRChem API

**Signature:** `FAIRChemCalculator()` raises about missing `task_name`.
**Cause:** UMA-style multi-task model; need `task_name="omat"` etc.
**Fix:** Check `FAIRChemCalculator.__init__` signature in the installed pkg.
Don't guess task name — read the signature.

**Signature:** eSEN env passes `task_name="..."` and crashes with
`unexpected keyword argument` or wrong head selected.
**Cause:** eSEN is single-task; task is baked into the checkpoint name
(e.g., `*-omol`, `*-oc25`). Don't pass `task_name`.
**Fix:** Use `FAIRChemCalculator(predictor)` for eSEN.

**Signature:** `KeyError` on the model name during
`pretrained_mlip.get_predict_unit(...)`.
**Cause:** Checkpoint name not in the registry (typo, or registry version
moved).
**Fix:** Catch and inspect the exception text — fairchem's error usually
lists valid names.

---

## Modal image / volume mount

**Signature:** `Runner failed with exception: cannot mount volume on
non-empty path: "/cache"` at function start, before any user code runs.
**Cause:** Cache env vars (`XDG_CACHE_HOME`, `HF_HOME`, etc.) baked into the
image *before* `uv_pip_install`, so uv wrote its own cache to `/cache`
during image build. The mount path is non-empty when the function starts.
**Fix:** Don't `.env(CACHE_ENV)` on the image. Set cache env vars at
function runtime — pass them to the probe subprocess via `env=`. The image's
own scratch stays under `/root/.cache` (uv default), and the mount path is
empty when Modal mounts the volume there.

---

## Worker / IPC

**Signature:** Local `modal run` log shows the test's "Testing X..." print
but nothing for many minutes; container is running.
**Cause:** Worker subprocess output is captured / buffered; main function
prints nothing while waiting on socket.
**Fix:** First, run `probe_env` (no-IPC) — does the env work standalone?
Second, open the Modal dashboard URL printed at run start — container
stderr/stdout streams there in realtime.

**Signature:** `RootstockCalculator` hangs on `__enter__`.
**Cause:** Worker subprocess died during import or model load; server is
waiting for a socket connection that won't come.
**Fix:** Reproduce with probe_env. If probe is fine, the IPC is broken — add
a worker-side stage marker print and a timeout in server.py.

---

## OOM / GPU

**Signature:** `CUDA out of memory` on first inference.
**Cause:** A10G has 24GB; large materials systems + heavy models exceed it.
**Fix:** Use a smaller probe system (single unit cell, not a supercell).
Bump to `gpu="A100"` for production smoke tests if needed.

---

## ASE API migrations

**Signature:** `ImportError: cannot import name 'ExpCellFilter' from 'ase.constraints'`
**Cause:** ASE 3.23 moved `ExpCellFilter` from `ase.constraints` to `ase.filters`. Libraries
pinned against older ASE (matgl, some FAIRChem versions) still import from the old path.
**Fix:** Monkeypatch before importing the offending library:
```python
import ase.constraints
if not hasattr(ase.constraints, "ExpCellFilter"):
    from ase.filters import ExpCellFilter
    ase.constraints.ExpCellFilter = ExpCellFilter
```

---

## DGL / torchdata compatibility

**Signature:** `ModuleNotFoundError: No module named 'torchdata.datapipes'` deep inside a DGL import.
**Cause:** DGL 2.x imports `torchdata.datapipes` (via `dgl.graphbolt`) at `__init__` time,
but `torchdata >= 0.7` removed `datapipes`. The packages co-resolve fine but fail at runtime.
**Fix:** Stub the entire `dgl.graphbolt` subpackage before `import dgl` runs. matgl only uses
DGL for graph construction — graphbolt is never called at inference time.
```python
import sys, types
for _name in [
    "dgl.graphbolt", "dgl.graphbolt.base", "dgl.graphbolt.dataloader",
    "dgl.graphbolt.feature_fetcher", "dgl.graphbolt.minibatch_transformer",
]:
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
```

---

## matgl model registry / loading

**Signature:** `KeyError` or `ValueError` when calling `matgl.load_model("M3GNet-MP-2021.2.8-PES")`
or `matgl.load_model("TensorNet-MatPES-PBE-v2025.1-PES")`.
**Cause:** matgl 1.0.0 dropped M3GNet-MP and the old TensorNet pretrained registry entries
(they moved to HuggingFace under `materialyze`). `load_model` with a bare name only checks
the GitHub manifest, which no longer lists these models.
**Fix:** Use `huggingface_hub.snapshot_download(repo_id="materialyze/<model>")` to get a local
path, then pass that path to `matgl.load_model(local_path)`.

**Signature:** `TypeError: Normalizer' is not in safe_globals` or similar during `matgl.load_model`.
**Cause:** PyTorch 2.6 changed `torch.load` default to `weights_only=True`; matgl's serialized
models include custom classes that must be explicitly allowlisted.
**Fix:**
```python
from matgl.data.transformer import Normalizer
torch.serialization.add_safe_globals([Normalizer])
```
Call this before `matgl.load_model`.

**Signature:** `pip install matgl` resolves but then `from matgl.apps._pes_pyg import ...` fails
at runtime (ModuleNotFoundError).
**Cause:** matgl 1.0.0 on PyPI doesn't include the `_pes_pyg` module needed for the
HuggingFace TensorNet models. Requires matgl 2.x from the git main branch.
**Fix:** Install matgl from git with `--no-deps` (its `lightning` dep can't resolve via uv):
```
uv pip install --system --no-deps "matgl @ git+https://github.com/materialsvirtuallab/matgl.git"
```
Then install the actual runtime deps separately: `torch`, `ase`, `pymatgen`, `monty`,
`ruamel.yaml`, `scipy`, PyG packages.

---

## matgl backend split (DGL vs PyG)

**Signature:** `ValueError: Invalid backend` when calling `matgl.set_backend("dgl")`.
**Cause:** matgl 2.x (from git main or PyPI >=2.0) dropped DGL and is PyG-only.
matgl 1.x (PyPI) is DGL-based. The two versions co-exist under the same package name.
**Fix:** The old DGL-based models (M3GNet-MP-2021.2.8-PES) are no longer accessible
via any current matgl version — the pretrained weights URL was removed. Use TensorNet
or CHGNet as substitutes. For new PyG-based TensorNet models use matgl 2.x (git).

**Signature:** `ValueError: No valid model found in pre-trained_models at
https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/`.
**Cause:** matgl 1.x PyPI fetches pretrained weights from the matgl GitHub repo,
but the `pretrained_models/` directory was removed (models moved to HuggingFace).
This affects M3GNet-MP-2021.2.8-PES and other legacy models.
**Fix:** No fix — the weights are gone from this path. Use materialyze HF models
for TensorNet, or use CHGNet/Orb as M3GNet-PES substitutes.

---

## fairchem API version split

**Signature:** `KeyError: '<model_name>'` when calling
`pretrained_mlip.get_predict_unit("<old-ocp-model>")` in fairchem>=2.0.
**Cause:** fairchem 2.0 is UMA-only; all OC20-era models (GemNet-OC, GemNet-T,
EquiformerV2, SCN, eSCN, PaiNN, SchNet, DimeNet++) were dropped from the 2.0 API.
**Fix:** Use `fairchem-core>=1.0.0,<2.0.0` and `OCPCalculator` with
`model_name_to_local_file` for auto-download. Checkpoint names follow the pattern
`GemNet-OC-Large-S2EF-OC20-All+MD` — inspect the ValueError message for the full list.

**Signature:** `FileNotFoundError: [Errno 2] No such file or directory: '<model_name>'`
when passing a model name string to `OCPCalculator(checkpoint_path=model)`.
**Cause:** fairchem 1.x `OCPCalculator` expects a local file path, not a model name.
**Fix:** Use `model_name_to_local_file(model, local_cache=cache_dir)` to download the
checkpoint first, then pass the returned path to `OCPCalculator`.

---

## ASE adsorbate molecule in probe

**Signature:** `KeyError: 'CO'` when calling `add_adsorbate(slab, "CO", ...)`.
**Cause:** `add_adsorbate` treats a string argument as an element symbol, not a
molecule name. "CO" is not in ASE's atomic number table.
**Fix:** Pass an `Atoms` object instead: `add_adsorbate(slab, molecule("CO"), ...)`.

---

## torchmd-net lightning dependency

**Signature:** `uv pip install torchmd-net` fails: "Because there are no versions of
lightning and all versions of torchmd-net depend on lightning...".
**Cause:** `lightning` (the PyPI package) has a dependency cycle or namespace conflict
that prevents uv from resolving it, even though the package itself works fine.
**Fix:** Same approach as matgl — install with `--no-deps` and list runtime deps
(torch, ase, torch-geometric, torch-scatter, torch-sparse, torch-cluster) separately.

---

# AMD / ROCm (droplet_workshop, targeting Frontier)

## pytorch-triton-rocm not found on PyPI

**Signature:** `No solution found ... Because there is no version of
pytorch-triton-rocm==3.5.1 and torch==2.9.1+rocm6.4 depends on
pytorch-triton-rocm==3.5.1, we can conclude that your requirements are
unsatisfiable.`
**Cause:** ROCm torch depends on `pytorch-triton-rocm`, which is published
*only* on the PyTorch ROCm index, never on PyPI. With `explicit = true` on the
index, only packages named in `[tool.uv.sources]` are drawn from it — so the
transitive triton dep was looked up on PyPI and not found. A `[tool.uv.sources]`
entry alone does NOT fix this: uv only applies sources to packages that appear
in `dependencies`.
**Fix:** List `pytorch-triton-rocm` as a *direct* dependency in the PEP 723
block AND map it in `[tool.uv.sources]`. No CUDA analog: `triton` is on PyPI,
so this failure cannot occur on NVIDIA.

## Index shadowing silently downgrades packages (DANGEROUS)

**Signature:** No error at all. The env builds, but resolves e.g.
`torchmetrics==1.0.3` instead of `1.9.0`.
**Cause:** "Fixing" the triton failure by dropping `explicit = true` makes the
ROCm index a general-purpose index. It mirrors an old subset of PyPI, and uv
takes each package from the first index that carries it — so unrelated packages
silently resolve to stale versions. This produces a *working* env running
different library code: the worst class of bug for reproducible science.
**Fix:** Keep `explicit = true`. Never relax index isolation to fix a missing
package; add the package as a direct dep instead (see above).

## Lockfile install can't find pinned versions on the ROCm index

**Signature:** `Because there is no version of filelock==3.29.7 and you require
filelock==3.29.7 ...` — during `uv pip install -r <lock>`, after `uv export`
succeeded. Hint text mentions `--index-strategy unsafe-best-match`.
**Cause:** `uv export --script` honors the PEP 723 index rules, but the emitted
lockfile carries only pinned versions. At install time uv again takes each
package from the first index that carries it *at all* — and the ROCm index
mirrors an old `filelock`/`sympy`/etc., which don't match the pins.
**Fix:** Pass the config's index URLs plus `--index-strategy unsafe-best-match`
to the install. Safe here precisely because the lock pins exact versions that
were already resolved under the strict `explicit` rules.

## ROCm torch wheel is ~4.2 GB (15+ GB installed)

**Signature:** `No space left on device (os error 28)` mid-install, often as
`Failed to clone .../torch/lib/libaotriton_v2.so`.
**Cause:** ROCm torch fat-binaries kernels for many gfx targets and bundles the
ROCm math libraries: ~2x the CUDA wheel. One venv is ~15-17 GB installed.
**Fix:** Put the venv root AND the uv cache on a large volume, not the
container/boot disk. Budget ~15-17 GB per env plus ~20 GB of shared wheel cache.

## torch-scatter / torch-sparse on ROCm: no wheels, must source-build

**Signature:** uv/pip cannot resolve `torch-scatter` on ROCm; `data.pyg.org`
serves only `+cuXXX` wheels (any rocm URL there 404s/403s).
**Cause:** The PyG project publishes CUDA wheels only — their build matrix
(torch x python x CUDA) already explodes, and ROCm isn't in it. AMD ships fat
ROCm wheels for *torch*, but nobody ships them for the third-party C++
extensions.
**Fix:** Source-build. VERIFIED WORKING on MI300X — full recipe below. The
resulting .so carries both gfx90a and gfx942, so ONE artifact serves Frontier
(MI250X) and MI300X. You do NOT need an MI250X to compile for one.

    # 1. Toolchain must MATCH the torch wheel's ROCm major.minor.
    #    A stock ROCm 5.7 hipcc CANNOT compile torch-2.9/rocm6.4 sources.
    apt install hipcc hip-dev rocm-llvm rocm-device-libs rocm-core  # from repo.radeon.com/rocm/apt/6.4
    export ROCM_PATH=/opt/rocm-6.4.0 HIP_PATH=/opt/rocm-6.4.0
    export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH

    # 2. Build for BOTH arches: Frontier's MI250X and the build box's MI300X.
    export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

    # 3. Install from GIT, not PyPI (see warp-mask entry below).
    pip install --no-build-isolation \
      "torch-scatter @ git+https://github.com/rusty1s/pytorch_scatter.git"

Verify with `check_isa.py <env> --require gfx90a`, and confirm the GPU op is
actually correct — not merely importable:

    torch_scatter.scatter_add(torch.tensor([1.,2.,3.,4.], device="cuda"),
                              torch.tensor([0,0,1,1],   device="cuda"))
    # -> [3.0, 7.0]

## torch-scatter PyPI release assumes 32-lane CUDA warps (breaks on AMD)

**Signature:** Source build fails with
`static assertion failed due to requirement 'sizeof(unsigned int) == 8':
The mask must be a 64-bit integer.` in
`amd_warp_sync_functions.h`, from `__shfl_up_sync<unsigned int, __half>`
instantiated at `csrc/hip/../hip/utils.cuh:13`.
**Cause:** A HARDWARE MODEL MISMATCH, not packaging. NVIDIA executes in warps
of 32 threads, so a warp-shuffle's lane mask is a 32-bit int. AMD executes in
wavefronts of 64 threads, so the mask must be 64-bit. torch-scatter's released
source hardcodes CUDA's `unsigned int` mask; ROCm 6.4 added a static_assert
that rejects it. (ROCm 5.7 didn't assert, but failed differently — ambiguous
`__shfl_up` overloads.)
**Fix:** torch-scatter's **git main already fixes this** and the fix is
UNRELEASED on PyPI:

    #ifdef USE_ROCM
      using warp_mask_t = unsigned long long;   // 64-lane wavefront
    #else
      using warp_mask_t = unsigned int;         // 32-lane warp
    #endif

So `pip install torch-scatter` is broken on ROCm while the repo is fine. Always
install these PyG extensions from git on AMD. Expect the same class of bug in
any CUDA kernel that hardcodes `0xffffffff` as a warp mask.

## torch-sparse spmm SILENTLY COMPUTES WRONG NUMBERS on ROCm (do not ship)

**Signature:** None. No error, no warning. It builds, imports, and returns
plausible-looking numbers that are WRONG.

    dense reference  : [[2, 2, 2], [7, 7, 7]]
    torch_sparse CPU : [[2, 2, 2], [7, 7, 7]]   correct
    torch_sparse GPU : [[2, 2, 2], [2, 2, 2]]   WRONG
    random 64x64 spmm: max|GPU - dense| = 12.47

**Cause:** The deeper half of the warp-width problem, and it is SEMANTIC, not a
type error. Fixing the 32->64-bit mask type (see the warp-mask entry above) only
silences the compiler. torch-sparse's `spmm` kernel is *algorithmically built
around a 32-lane warp*: it has a warp cooperatively reduce one row of the sparse
matrix via shuffles, looping over lanes. AMD wavefronts are **64** lanes, so the
reduction spans the wrong thread set and drops contributions — row 1 above lost
its second nonzero (3+4=7 became 2).

**THE TRAP:** patch-until-it-compiles produces a working-looking install that
corrupts the physics. The probe CANNOT catch this — energies and forces still
come out, they are just wrong. Only a CPU-vs-GPU numerical comparison catches it.

**Fix:** Do NOT ship torch-sparse on ROCm on the strength of a successful build.
Either (a) confirm the model never calls `torch_sparse.spmm` (many MLIPs pull
torch-sparse in as a transitive dep of PyG but only ever use torch-scatter), or
(b) rewrite the spmm kernel for a 64-lane wavefront (`warpSize` is not 32 on
AMD), or (c) force the CPU/fallback path for spmm.

**ALWAYS numerically validate a source-built GPU extension against CPU.** Not
"does it import", not "does the probe pass" — compare the numbers:

    cpu = op(src, idx)
    gpu = op(src.cuda(), idx.cuda()).cpu()
    assert (cpu - gpu).abs().max() < 1e-4

Verified on MI300X / torch 2.9.1+rocm6.4:
  torch-scatter  CORRECT (scatter_add/mean/max, segment_csr; max err ~5e-6)
  torch-cluster  CORRECT (radius_graph edge-identical to CPU)
  torch-sparse   WRONG   (spmm)

## gfx942-only extension: works on MI300X, dies on Frontier

**Signature:** Extension builds and probes fine on the MI300X box, then on
Frontier: `HSA_STATUS_ERROR_INVALID_ISA`, "invalid device function", or a
segfault at first kernel launch.
**Cause:** A source build targets only the build host's arch (gfx942) unless
told otherwise. Official torch wheels are unaffected — they are fat binaries
carrying gfx90a (verified: 23/23 fat libs in the mace env).
**Fix:** `PYTORCH_ROCM_ARCH="gfx90a;gfx942"` before building, then gate on it —
the probe CANNOT catch this, since the code runs fine on the build box:

    python3 check_isa.py <env> --require gfx90a

## CUDA torch wheel resolved on a ROCm box

**Signature:** `torch.cuda.is_available()` returns False on a machine where
`rocm-smi` shows the GPU; or import errors mentioning `libcudart.so` /
`libnvrtc`.
**Cause:** The config resolved torch from plain PyPI (a CUDA build) instead of
the ROCm index.
**Fix:** Add to the PEP 723 block: `[tool.uv.sources] torch = { index = "pytorch-rocm" }`
plus `[[tool.uv.index]] name = "pytorch-rocm" url = "https://download.pytorch.org/whl/rocm6.4"
explicit = true`. Note `--device cuda` is correct on ROCm — HIP devices are
exposed under the `cuda` API; only the wheel source changes.

## No ROCm wheels for torch-scatter / torch-sparse / torch-cluster

**Signature:** `uv export` / install fails resolving `torch-scatter` against the
rocm index, or only `+cuXXX` wheels are found on data.pyg.org.
**Cause:** data.pyg.org publishes CUDA wheels only; there is no official ROCm
build of the PyG C extensions.
**Fix (in order):** (1) Try dropping torch-scatter/sparse entirely — PyG >= 2.3
falls back to pure-torch scatter ops and many models work without them.
(2) Source-build on the box with `PYTORCH_ROCM_ARCH="gfx90a;gfx942"` (both
Frontier MI250X and droplet MI300X) and install with `--no-deps`.

## gfx arch mismatch between droplet and Frontier

**Signature:** Works on the MI300X droplet, but on MI250X:
`HSA_STATUS_ERROR_INVALID_ISA`, "invalid device function", or instant
segfault at first kernel launch.
**Cause:** A source-built extension was compiled only for gfx942 (MI300X);
Frontier's MI250X is gfx90a. Official torch ROCm wheels are multi-arch and
unaffected — only locally-built extensions hit this.
**Fix:** Rebuild with `PYTORCH_ROCM_ARCH="gfx90a;gfx942"`.

## cuEquivariance / CUDA-only acceleration packages

**Signature:** Install failure or import error for `cuequivariance*` on ROCm.
**Cause:** NVIDIA-only package.
**Fix:** Omit it. MACE and friends run on the pure e3nn/torch path (slower,
but the configs never enabled acceleration anyway).

## MIOpen cache writes at first inference

**Signature:** First inference is slow or errors with MIOpen unable to write
its cache in a read-only HOME.
**Cause:** MIOpen (ROCm's cuDNN analog) writes kernel-tuning caches under
`$HOME/.cache/miopen` — the ROCm analog of the CUDA_CACHE_PATH redirect
rootstock already does.
**Fix:** On the droplet the workshop's HOME redirect covers it. On Frontier,
set `MIOPEN_USER_DB_PATH` / `MIOPEN_CACHE_DIR` to a writable per-user dir
alongside rootstock's existing cache redirection.

---

## How to read this file when porting a new MLIP

1. Try the build / probe.
2. If it fails, grep the error text against signatures above.
3. If no match, add the new signature here once you've root-caused it.

The taxonomy compounds: every new env we port should leave behind one or
more entries that cut iteration time for the next one.
