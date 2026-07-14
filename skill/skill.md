---
name: rootstock
description: Use this skill whenever the user wants to run a machine-learning interatomic potential (MLIP) — MACE, UMA, eSEN, ORB, TensorNet, or any other foundation potential carried by the Rootstock registry — on an HPC cluster. Trigger on any mention of Rootstock, RootstockCalculator, or the Rootstock dashboard, or on requests like "compute an adsorption energy on Perlmutter", "relax this structure with MACE on Perlmutter", or "what MLIPs are available on this cluster". Also trigger when the user asks for catalysis, materials, or molecular MLIP calculations on a named HPC cluster, even if they don't say "Rootstock". The skill covers (a) discovering which models are deployed where, and (b) configuring `RootstockCalculator` correctly for a given cluster + checkpoint.
---

# Rootstock

Rootstock runs MLIPs on HPC clusters from a single ASE-compatible `RootstockCalculator`. Each MLIP family lives in its own pre-built Python environment on a shared filesystem; swapping potentials is a one-line change in the user's code.

Use this skill in two phases:

1. **Discover** what's deployed where, from the live dashboard manifest.
2. **Call** `RootstockCalculator` with a `(cluster, checkpoint)` pair.

Getting Python code onto a compute node (job submission, file staging, credentials) is out of scope here — see "Getting code onto a cluster" below.

---

## Phase 1 — Discover what's available

The Rootstock admin dashboard exposes a JSON dump at:

    https://garden-ai-prod--rootstock-admin-dashboard.modal.run/

**Fetch it with `curl` or Python `requests`, NOT `WebFetch`.** The dump is ~180 KB of JSON; `WebFetch` summarizes responses through a small model and silently drops entries, including foundation-model envs (UMA in particular has been observed to disappear from `WebFetch` summaries). Use one of:

```bash
curl -sS https://garden-ai-prod--rootstock-admin-dashboard.modal.run/ | jq ...
```

```python
import requests
dump = requests.get("https://garden-ai-prod--rootstock-admin-dashboard.modal.run/").json()
```

Then process the raw JSON yourself. GET-ing the URL returns `{"manifests": [...]}`, one manifest per cluster:

```json
{
  "schema_version": 3,
  "cluster": "perlmutter",
  "root": "/global/cfs/cdirs/m4845/rootstock",
  "cache_root": null,
  "rootstock_version": "0.9.0",
  "python_version": "3.11",
  "last_updated": "2026-05-13T20:12:43+00:00",
  "maintainer": {"name": "...", "email": "..."},
  "environments": {
    "uma": {
      "status": "ready",
      "built_at": "2026-05-05T20:13:31+00:00",
      "python_requires": ">=3.10,<3.11",
      "dependencies": {"fairchem-core": "...", "torch": "...", "ase": "..."},
      "source": "# /// script\n# requires-python = ...\n...def setup(...):\n  ...",
      "source_hash": "sha256:...",
      "checkpoints": {
        "uma-s-1p1": {
          "fetched_at": "...",
          "verified_at": "...",
          "verified_device": "cuda",
          "last_error": null
        }
      }
    }
  }
}
```

Older manifests may report `schema_version` as the string `"1"` with a different env-entry shape. Filter out non-`3` manifests, or treat them as "stale, ask the maintainer to refresh."

To answer "is checkpoint X available on cluster Y?":

```python
import requests

dump = requests.get("https://garden-ai-prod--rootstock-admin-dashboard.modal.run/").json()
cluster = next(m for m in dump["manifests"] if m["cluster"] == "perlmutter")

for env_name, env in cluster["environments"].items():
    for ckpt_id, ckpt in env.get("checkpoints", {}).items():
        ok = ckpt.get("verified_at") and not ckpt.get("last_error")
        print(f"{env_name:12s} {ckpt_id:40s} {'OK' if ok else 'STALE/FAIL'}")
```

Things to read off the manifest, in order of importance:

- **`environments[env].status == "ready"`** — the venv itself built. Anything else and nothing in that env will work.
- **`checkpoints[id].verified_at`** vs **`environments[env].built_at`** — if the env was rebuilt more recently than the checkpoint was verified, treat the checkpoint as stale until the next nightly smoke test.
- **`checkpoints[id].last_error`** — non-null means the most recent verification failed. The error text usually tells you why.
- **`dependencies`** — exact package versions in the deployed env, useful for matching driver-side code to what's on the cluster.
- **`source`** — the PEP-723 source file of the environment, including the `setup()` signature. This is the ground truth for which `setup_kwargs` an env accepts.

If the cluster isn't in the dump, Rootstock isn't deployed there. The set of clusters the dump carries changes over time — read it from the manifest, don't assume a fixed list.

**Env-name aliases.** Some clusters carry both a bare env name (`uma`) and a legacy `_env`-suffixed alias (`uma_env`) from an older naming scheme. The bare names hold the real checkpoints; the `_env` aliases are usually empty stubs. Ignore the aliases and use the bare names.

---

## Phase 2 — Call `RootstockCalculator`

Rootstock itself is a small Python package; the heavy MLIP environments live at the cluster's shared `root` and are looked up by name. On a compute node with the install root visible:

```python
from ase.build import bulk
from rootstock import RootstockCalculator

atoms = bulk("Cu", "fcc", a=3.6) * (4, 4, 4)

with RootstockCalculator(
    cluster="perlmutter",
    checkpoint="mace-mp-0-medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
```

Constructor parameters:

| Parameter      | Type | Notes                                                                                                       |
|----------------|------|-------------------------------------------------------------------------------------------------------------|
| `checkpoint`   | str  | Canonical id from the manifest, e.g. `"mace-mp-0-medium"`, `"uma-s-1p1"`. Required.                          |
| `cluster`      | str  | Known cluster name, e.g. `"perlmutter"` — use whatever names the manifest carries. Mutually exclusive with `root`. |
| `root`         | str  | Custom install root for clusters not in the registry. Mutually exclusive with `cluster`.                     |
| `cache_root`   | str  | Override the model-weight cache. Defaults to the cluster's registered cache.                                 |
| `device`       | str  | `"cuda"` (default) or `"cpu"`.                                                                              |
| `setup_kwargs` | dict | Forwarded to the env's `setup()`. Cannot include `checkpoint` or `device`. The env's `source` lists what it accepts. |

The hosting env is resolved automatically from the checkpoint id — pick a checkpoint, Rootstock finds the env that declares it.

**Always use the context manager.** It tears down the worker subprocess. Without it, the worker can leak and pin a GPU.

**Reuse the calculator for multiple `atoms`.** Each `with` block spawns a worker and pays a 5–30 s model load. Do all the calculations you can inside one block:

```python
with RootstockCalculator(cluster="perlmutter", checkpoint="mace-mp-0-medium") as calc:
    for atoms in structures:
        atoms.calc = calc
        results.append((atoms.get_potential_energy(), atoms.get_forces()))
```

---

## Picking a model

There is no built-in default. Which potential to use is the user's call, bounded by what the manifest shows as deployed and verified on the target cluster.

- If the user names a model, use it — confirm it's in the manifest for that cluster.
- If the user doesn't name one, choose based on the system and what's available, and state which you picked and why. When it's genuinely ambiguous, ask rather than guess.
- The manifest is the list of what's actually available. Don't assume a model is deployed without checking.

Many envs accept extra `setup_kwargs` beyond `checkpoint` and `device`. The authoritative list for any env is its `setup()` signature, embedded in the manifest's `source` field — read it before passing kwargs.

Some models select a task-specific head via `setup_kwargs` — UMA, for example, takes a `task`. The head should match the system class, and choosing it wrong silently degrades the energy. Which values are valid, and which fits the system, is model-specific; read the env's `source` for the accepted kwargs, and the model's own documentation for what to pass.

---

## Getting code onto a cluster

Out of scope for this skill, but the runtime just needs (a) `rootstock` importable in the Python the job runs, and (b) read access to the cluster's install root. Common paths: `ssh` + `sbatch`/`qsub`, a user-deployed Globus Compute endpoint, a facility JupyterHub, or the IRI Facility API. The dispatch step varies by cluster and by user; `ssh` + `sbatch` is usually the easiest to drive from an external agent.

---

## Common pitfalls

- **Login nodes don't have GPUs.** Worker startup fails with a CUDA error. Either run on a compute node, or pass `device="cpu"` (much slower).
- **Login nodes often have no outbound network.** Affects deployment (`rootstock add` fetches checkpoints), not normal use.
- **First call inside a fresh worker is slow.** Model load dominates wall time for short jobs. Batch into one `with` block.
- **`last_error != null` in the manifest is real.** If the dashboard shows a checkpoint as broken on a cluster, it's broken. Pick a different one or a different cluster.
- **Env Python versions differ.** UMA pins `>=3.10,<3.11`; MACE and TensorNet differ. Driver code can run on any Python — Rootstock fires up the right interpreter in the worker — but don't try to share live Python objects across the boundary.
