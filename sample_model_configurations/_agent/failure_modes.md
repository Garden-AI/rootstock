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

## How to read this file when porting a new MLIP

1. Try the build / probe.
2. If it fails, grep the error text against signatures above.
3. If no match, add the new signature here once you've root-caused it.

The taxonomy compounds: every new env we port should leave behind one or
more entries that cut iteration time for the next one.
