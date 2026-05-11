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

---

## HuggingFace auth & cache

**Signature:** `OSError: ... 401 Client Error ...` or `gated repo` on first
model download.
**Cause:** `HF_TOKEN` not set in container, or token lacks gated-repo
permission, or model license not accepted.
**Fix:** Add `secrets=[modal.Secret.from_name("huggingface-token")]` to the
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

## How to read this file when porting a new MLIP

1. Try the build / probe.
2. If it fails, grep the error text against signatures above.
3. If no match, add the new signature here once you've root-caused it.

The taxonomy compounds: every new env we port should leave behind one or
more entries that cut iteration time for the next one.
