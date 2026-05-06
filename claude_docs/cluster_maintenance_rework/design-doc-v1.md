# Rootstock: First-Class Checkpoints, Verification, and Calculator Kwargs

**Status:** Draft for implementation
**Audience:** Claude Code
**Author:** Claude (with Will)
**Target version:** v0.8.0

---

## 1. Motivation

The current `rootstock install` builds an isolated virtual environment and optionally pre-downloads model weights via `--models foo,bar`, which calls `setup(model, "cpu")` once. This has two problems we hit recently when an intern tried to bring up Rootstock on a new cluster:

1. **"Installed" does not mean "works on GPU."** The pre-download step runs on the login node on CPU. A checkpoint that downloads cleanly there can still fail on the GPU node. There is no command that re-validates everything on the actual hardware users will run on.
2. **The manifest's view of checkpoints is opaque.** `EnvironmentInfo.checkpoints: list[str]` records names but no metadata: when they were fetched, whether they were verified, on what device, with what error if any. The dashboard renders the list but it carries no signal about what is currently usable.

A related, smaller pain point: UMA's task head (`task_name="omat"`) is hardcoded inside `uma_env.py`. Selecting a different head requires editing the env file.

This design addresses all three by:

- Letting `RootstockCalculator` forward an explicit `setup_kwargs` dict to the env's `setup()` function.
- Making manifest checkpoint entries first-class records with timestamps and last-error fields.
- Adding `rootstock add` (download + verify, idempotent) and `rootstock smoke-test` (re-verify everything).

---

## 2. Goals

- Reduce uncertainty about what is installed *and currently working* on a given cluster.
- Make GPU verification explicit, repeatable, and fast enough to run as a nightly cron.
- Allow users to vary kwargs like UMA's `task` without modifying env files or the calculator.
- Keep the maintainer workflow (login node CPU + restricted GPU nodes) ergonomic via an idempotent download-or-verify command.

## 2a. Non-goals

- Bundling kwargs into named manifest entries / "presets". Kwargs are passed at call time. The manifest does not record kwargs at all.
- Multi-state staleness modeling. A checkpoint either has a current `verified_at > env.built_at` or it doesn't. Env rebuilds invalidate prior verifications and require a re-verify.
- Auto-discovering on-disk weight paths. Where each library caches weights is human-curated knowledge that lives in the project Almanac.
- C++ `fix_rootstock` kwargs forwarding. The Python `serve` command will accept kwargs; the LAMMPS-side parser update is a follow-on.
- Per-environment custom test atoms. The smoke test uses one hardcoded structure (see §6).

---

## 3. Calculator kwargs passthrough

### 3.1 User-facing API

`RootstockCalculator.__init__` gains a single `setup_kwargs: dict | None = None` parameter. Anything in that dict is forwarded as keyword arguments to the env's `setup()` function in the worker process. There is no loose-kwarg sweep — kwargs not in `setup_kwargs` are passed to `super().__init__()` as ASE expects.

```python
with RootstockCalculator(
    cluster="della",
    model="uma",
    checkpoint="uma-s-1p1",
    device="cuda",
    setup_kwargs={"task": "omol"},
) as calc:
    ...
```

The reasoning for the explicit channel: ASE's `Calculator.__init__` accepts `label`, `atoms`, `directory`, `restart`, `ignore_bad_restart_file`. A loose-kwarg sweep would either steal those or have to maintain a denylist. Making `setup_kwargs` explicit avoids the ambiguity entirely.

Validation in `__init__`:
- If `setup_kwargs` contains the keys `model` or `device`, raise `TypeError` — those are reserved and must be passed at the top level.

### 3.2 Wrapper / worker plumbing

The setup-kwargs need to reach the worker subprocess. The current wrapper template is a `.format()`-substituted Python source string with `model` and `device` baked in as quoted literals — extending this with arbitrary Python values would force us to round-trip through `repr()` and worry about escaping. Avoid that by writing kwargs to a temp JSON file alongside the wrapper temp file.

**`rootstock/environment.py` — wrapper template:**

```python
WRAPPER_TEMPLATE = """
import sys, json
sys.path.insert(0, "{env_dir}")
from env_source import setup
from rootstock.worker import run_worker

with open("{kwargs_path}") as f:
    setup_kwargs = json.load(f)

run_worker(
    setup_fn=setup,
    model="{model}",
    device="{device}",
    socket_path="{socket_path}",
    setup_kwargs=setup_kwargs,
)
"""
```

`EnvironmentManager.generate_wrapper()` gains a `setup_kwargs: dict` parameter, writes it to a sibling temp file, registers both temp files for cleanup, and substitutes the path into the template.

**`rootstock/worker.py` — `run_worker` signature:**

```python
def run_worker(
    setup_fn,
    model: str,
    device: str,
    socket_path: str,
    setup_kwargs: dict | None = None,
    log=None,
):
    setup_kwargs = setup_kwargs or {}
    if log:
        print(f"[Worker] Calling setup({model!r}, {device!r}, **{setup_kwargs!r})", file=log, flush=True)
    calculator = setup_fn(model, device, **setup_kwargs)
    ...
```

**`rootstock/server.py` — `RootstockServer`** gains a `setup_kwargs` field, plumbed from the calculator and forwarded to `generate_wrapper()`.

**Failure mode:** If the env's `setup()` doesn't accept the kwargs (e.g., user passed `setup_kwargs={"task": "omat"}` to `mace_env`), the worker raises `TypeError` during model load. The server-side accept loop already reports worker death via stdout/stderr. This is the loud failure we want.

### 3.3 Env file changes

Only `uma_env.py` needs editing — to expose the `task` kwarg:

```python
def setup(model: str = "uma-s-1p1", device: str = "cuda", task: str = "omat"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    predictor = pretrained_mlip.get_predict_unit(model, device=device)
    return FAIRChemCalculator(predictor, task_name=task)
```

`omat` is preserved as the default. No `**kwargs` swallowing — if a user passes an unknown kwarg, the `TypeError` from `setup()` propagating up is the right signal.

Other env files (`mace_env`, `chgnet_env`, `tensornet_env`, `esen_env`) are unchanged. Calling them with non-empty `setup_kwargs` will fail loudly, which is correct.

---

## 4. Manifest schema (v2)

### 4.1 New `CheckpointInfo`

`rootstock/manifest.py`:

```python
@dataclass
class CheckpointInfo:
    """Metadata for a single checkpoint registered with an environment."""
    fetched_at: str | None      # ISO8601, set when download succeeds
    verified_at: str | None     # ISO8601, set when smoke test passes
    verified_device: str | None # "cuda" or "cpu" (None if never verified)
    last_error: str | None      # most recent error from add or smoke-test (None if last attempt succeeded)

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointInfo": ...
```

The `name` is the dict key, not a field on the value.

`EnvironmentInfo.checkpoints` changes type:

```python
@dataclass
class EnvironmentInfo:
    ...
    checkpoints: dict[str, CheckpointInfo] = field(default_factory=dict)
```

### 4.2 "Currently verified" predicate

Not stored; computed:

```python
def is_verified(env: EnvironmentInfo, ckpt: CheckpointInfo) -> bool:
    if ckpt.verified_at is None:
        return False
    return ckpt.verified_at > env.built_at  # ISO8601 sorts lexically
```

Used by `status` and `smoke-test` for display. Lives in `rootstock/manifest.py`.

### 4.3 Schema version: hard cutover, throwaway migration

`Manifest.schema_version` goes from `1` to `2`. The new `from_dict` accepts only `schema_version == 2`. Anything else raises:

```
RuntimeError: manifest at <path> is schema_version=1, expected 2.
Run scripts/migrate_manifest_v1_to_v2.py against this manifest before continuing.
This script will be removed in a follow-up release.
```

A standalone script `scripts/migrate_manifest_v1_to_v2.py` does the conversion: reads the manifest, transforms `checkpoints: list[str]` into `checkpoints: {name: {fetched_at: None, verified_at: None, verified_device: None, last_error: None}}`, bumps `schema_version`, writes back. The maintainer runs this script manually on each cluster's `manifest.json` before deploying the new rootstock version.

**Both the script and the v1-rejection branch of `from_dict` are explicitly temporary.** Once known clusters have been migrated, a follow-up PR deletes:
- `scripts/migrate_manifest_v1_to_v2.py`
- The `schema_version != 2` rejection in `from_dict` (it just trusts the manifest after that)

Target window: ~1 week. Tracked via TODO comments in both files referencing a single follow-up issue.

The dashboard renderer in `docs/clusters.md` reads `env.checkpoints`. Update its JS to handle the new dict shape with verified/stale indicators. Brief breakage during rollout is acceptable.

---

## 5. *(deleted — see §6)*

---

## 6. Verification logic

A single helper in a new module `rootstock/verify.py`:

```python
def verify_checkpoint(
    root: Path,
    env_name: str,
    checkpoint: str,
    device: str,
    setup_kwargs: dict,
) -> tuple[bool, str | None]:
    """
    Run a single forward pass to verify a checkpoint.

    Returns (success, error_message). On success, error_message is None.
    """
```

The test atoms are hardcoded inside `verify.py` — one structure for everything:

```python
def _smoke_test_atoms() -> "ase.Atoms":
    from ase import Atoms
    atoms = Atoms(
        "H2O",
        positions=[[0.00, 0.00, 0.00],
                   [0.96, 0.00, 0.00],
                   [0.24, 0.93, 0.00]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    # eSEN OMol checkpoints expect these. Harmless for everyone else.
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    # Break any accidental symmetry so forces aren't trivially zero.
    atoms.positions[1, 1] += 0.05
    return atoms
```

This is goofy for inorganic-focused MLIPs (a water molecule in vacuum is out-of-domain for CHGNet or MACE-MP), but the smoke test only asks "did the computational pipeline produce a finite, non-degenerate result?" — not "is the answer chemically meaningful?". Out-of-domain models still return finite numbers; that's all we need.

Implementation outline:

1. Spawn a `RootstockServer` for `(env_name, checkpoint, device, setup_kwargs)`.
2. Connect, send INIT (with atomic numbers from `_smoke_test_atoms()`), POSDATA (positions, cell), GETFORCE.
3. On response, assert:
   - energy is finite (`np.isfinite(energy)`)
   - forces array shape matches `(n_atoms, 3)`
   - forces are finite (`np.all(np.isfinite(forces))`)
   - forces are not all (near-)zero: `np.linalg.norm(forces) > 1e-8`
   - virial is finite
4. Tear down worker.
5. Return `(True, None)` on full success, else `(False, str(exc))`.

The forces-not-all-zero check guards against the silent failure where a model returns zeros for everything. Cheap insurance.

Per-checkpoint runtime: should be well under 30s on GPU, dominated by model load. `smoke-test` runtime is linear in checkpoint count.

`verify_checkpoint` is the shared entry point used by both `add` (one checkpoint) and `smoke-test` (all of them).

---

## 7. CLI changes

### 7.1 New: `rootstock add`

```
rootstock add <env> <checkpoint> [--kwarg key=val ...] [--device DEVICE] [--no-verify]
                                 [--root PATH] [--no-push]
```

Idempotent download-or-verify. "Add this checkpoint to the install."

1. Resolve `env_source = root / "environments" / f"{env}_env.py"`. Error if missing.
2. Look up or create the manifest entry at `env.checkpoints[checkpoint]`.
3. **Download phase** (skipped if `entry.fetched_at is not None`):
   - Run `setup(checkpoint, "cpu", **kwargs)` in the env subprocess to trigger any cache-aware downloads. CPU is used regardless of `--device`, since download is the only goal and CPU is the lowest-common-denominator that works on login nodes.
   - On success: `entry.fetched_at = now()`. Clear `last_error`.
   - On failure: set `last_error`, save manifest, exit 1.
4. **Verify phase** (skipped if `--no-verify`):
   - Call `verify_checkpoint(root, env_name, checkpoint, device, kwargs)`.
   - On success: `entry.verified_at = now()`, `entry.verified_device = device`, clear `last_error`.
   - On failure: clear `verified_at` and `verified_device`, set `last_error`, save manifest, exit 1.
5. Save manifest. Push if configured (unless `--no-push`).

`--device` defaults to `cuda`. `--no-verify` is the login-node escape hatch.

`--kwarg key=val` is repeatable. Values are JSON-parsed first; on parse failure they fall back to strings. So `--kwarg task=omat` → `"omat"`, `--kwarg charge=-1` → `-1`, `--kwarg enabled=true` → `True`.

Examples:

```bash
# Login node: download only
rootstock add uma uma-s-1p1 --kwarg task=omat --no-verify

# GPU node, no network: skip download (already fetched), verify
rootstock add uma uma-s-1p1 --kwarg task=omat

# All-purpose node: do both
rootstock add mace medium

# Re-verify (fetched_at unchanged, verified_at refreshed)
rootstock add mace medium
```

A failure on either phase exits nonzero. The manifest is still saved with the error recorded so subsequent `status` shows what went wrong.

### 7.2 New: `rootstock smoke-test`

```
rootstock smoke-test [--env ENV] [--checkpoint CKPT] [--device DEVICE]
                     [--root PATH] [--json] [--no-push]
```

Re-verify checkpoints already in the manifest. Never downloads.

- No filters: test every `(env, checkpoint)` where the env is built and the checkpoint has `fetched_at != None`.
- `--env foo`: filter to that env.
- `--checkpoint bar`: filter further (requires `--env`).
- `--device`: defaults to `cuda`.
- `--json`: emit a machine-readable summary.

For each entry tested, run `verify_checkpoint` with `setup_kwargs={}`, update timestamps and `last_error` exactly as `add` does.

Exit code is 0 if all tested checkpoints passed, 1 otherwise. Suitable for cron:

```
0 4 * * * rootstock smoke-test --json > /var/log/rootstock-smoke.log 2>&1
```

Human output:

```
mace/medium      [PASS]  cuda  3.2s
mace/large       [PASS]  cuda  4.1s
uma/uma-s-1p1    [FAIL]  cuda  TypeError: ...
tensornet/...    [PASS]  cuda  2.7s

3 passed, 1 failed in 12.4s
```

**Note on kwargs:** `smoke-test` always passes `setup_kwargs={}`, relying on env defaults. A checkpoint that only works with non-default kwargs will appear failing in nightly smoke-test even though `add` succeeded. The remedy is to make the preferred kwargs the env's default. Documented in `smoke-test --help`.

### 7.3 Modified: `rootstock install`

Drop the `--models` argument entirely. `install` becomes login-node-only and concerns itself solely with the venv. `rootstock install foo.py --models a,b,c` should fail with a clear migration error message:

```
Error: --models has been removed. Use 'rootstock add' instead:
  rootstock add <env> <checkpoint>
```

Update help text and `README.md` accordingly.

### 7.4 Modified: `rootstock status`

Extend output to show per-checkpoint verification state:

```
mace_env:
  Status: ready
  Built: 2026-04-25T14:30:00Z
  Checkpoints (3):
    medium    fetched 2026-04-25  verified 2026-04-30 (cuda)  ✓
    large     fetched 2026-04-25  verified 2026-04-25 (cuda)  ⚠ stale (env rebuilt 2026-04-30)
    small     fetched 2026-04-25  not verified                ⚠
```

The "stale" indicator is computed from `is_verified()`, not stored. `--json` outputs the raw manifest data plus the computed `verified_current: bool` for convenience.

### 7.5 Modified: `rootstock serve`

`serve` already takes `--checkpoint` and `--device`. Add `--kwarg key=val` (repeatable, same JSON-parsing rule as `add`) and forward to the worker via the same `setup_kwargs` plumbing as the calculator. Keeps the LAMMPS-side Python aligned with the ASE-side Python.

The C++ `fix_rootstock` parser doesn't change for this PR — out of scope.

---

## 8. File-by-file changes

| File | Change |
|---|---|
| `rootstock/manifest.py` | Add `CheckpointInfo`. Change `EnvironmentInfo.checkpoints` to dict. Add `is_verified()`. Bump `schema_version` to 2. Reject non-v2 manifests in `from_dict` (temporary). |
| `scripts/migrate_manifest_v1_to_v2.py` | NEW. Standalone migration. **Marked for deletion** — TODO comment referencing the follow-up issue. |
| `rootstock/calculator.py` | Add `setup_kwargs` param. Validate no reserved keys. Pass to `RootstockServer`. |
| `rootstock/server.py` | Accept `setup_kwargs`, forward to `generate_wrapper`. |
| `rootstock/environment.py` | `generate_wrapper` accepts `setup_kwargs`, writes JSON sidecar, updates `WRAPPER_TEMPLATE`. Add JSON sidecar to `_temp_files` for cleanup. |
| `rootstock/worker.py` | `run_worker` accepts and forwards `setup_kwargs` to `setup_fn`. |
| `rootstock/verify.py` | NEW. `verify_checkpoint(...)` plus the hardcoded H2O test atoms. |
| `rootstock/commands/add.py` | NEW. Implements `cmd_add`. |
| `rootstock/commands/smoke_test.py` | NEW. Implements `cmd_smoke_test`. |
| `rootstock/commands/install.py` | Remove `--models` handling and the related download loop. |
| `rootstock/commands/status.py` | Render checkpoints dict. Compute `is_verified` per checkpoint. Add `--json`. |
| `rootstock/commands/serve.py` | Add `--kwarg` parsing, pass `setup_kwargs` through. |
| `rootstock/commands/manifest.py` | Update `_refresh_manifest_environments` to preserve dict-shaped `checkpoints`. |
| `rootstock/cli.py` | Wire up `add` and `smoke-test`. Make `--models` on `install` an error. Add `--kwarg` on `serve`. |
| `sample_model_configurations/.../uma_env.py` | Replace hardcoded `task_name="omat"` with `task="omat"` kwarg. |
| `docs/clusters.md` | Dashboard JS renders dict-shaped checkpoints with verified/stale indicators. |
| `docs/api.md`, `docs/cluster-setup.md`, `README.md` | Document new commands, `--models` removal, kwargs forwarding. |
| `CLAUDE.md` | Update workflow notes. |
| `tests/` | New tests for `verify_checkpoint`, kwargs plumbing in wrapper, CLI happy paths and idempotency for `add`. |

---

## 9. Test plan / acceptance criteria

### 9.1 Unit / integration tests

- **Wrapper kwargs round-trip**: `generate_wrapper` with non-empty kwargs produces a JSON sidecar that the worker reads and unpacks correctly. Booleans, ints, negative ints, strings all pass through.
- **Calculator validation**: passing `setup_kwargs={"model": "x"}` or `setup_kwargs={"device": "x"}` raises `TypeError`.
- **Verify happy path**: against a test env in CI, `verify_checkpoint` returns `(True, None)`.
- **Verify zeros-detector**: a stub setup function returning a calculator with all-zero forces is rejected.
- **CLI**: `rootstock install --models foo,bar` exits nonzero with the migration message.
- **Idempotency**: `rootstock add mace medium --no-verify` then `rootstock add mace medium` results in both `fetched_at` and `verified_at` being set on the second call.
- **Migration script**: a v1 manifest fixture is converted by the script to a valid v2 manifest that round-trips through `from_dict`.

### 9.2 Live deployment validation: Perlmutter fresh install

The point of this work is the Perlmutter fresh install. That is the acceptance test:

1. `rootstock init --cluster perlmutter` (after adding `"perlmutter"` to `CLUSTER_REGISTRY` — trivial one-liner, bundled with this PR).
2. `rootstock install` for each desired env on a login node.
3. `rootstock add <env> <ckpt> --no-verify` on a login node for each desired checkpoint (network-restricted GPU nodes assumed).
4. `rootstock add <env> <ckpt>` on a GPU node — no-op for download (`fetched_at` set), verifies on GPU.
5. `rootstock status` shows a clean grid of verified entries.

If any of those steps surface a usability snag, that's a finding and may motivate a follow-up. The whole point of this exercise is to flush out the next layer of papercuts on a fresh setup.

A `docs/cluster-setup.md` update walking through this exact sequence comes out of doing the install — write it after, not before.

---

## 10. Open questions / future work

- **Storing the kwargs used at last verification.** §7.2 deliberately punts. If too many checkpoints show as nightly-failing because of non-default kwargs, revisit.
- **`fix_rootstock` (LAMMPS) kwargs forwarding.** Add `kwarg key value` keyword pairs to the C++ parser and forward to `serve --kwarg`. Out of scope here.
- **Smoke-test parallelism.** Currently sequential. If runtime crosses ~10 minutes, parallelize across GPU IDs.
- **Almanac entries for weight cache locations.** Document where each library drops its weights. Documentation, not code.

---

## 11. Compatibility summary

| Item | Behavior |
|---|---|
| Old user code calling `RootstockCalculator(model="mace", checkpoint="medium")` | Works unchanged. |
| Old user code passing extra ad-hoc kwargs | Now goes to `super().__init__()` (ASE) — silently ignored unless ASE recognizes them. To reach `setup()`, must use `setup_kwargs={...}`. |
| Old `rootstock install foo.py --models a,b` | Errors with migration message. |
| Old manifest (v1, list-shaped checkpoints) | Errors on read; maintainer must run the migration script once before upgrading. |
| Existing custom env files with `def setup(model, device)` | Unchanged unless caller passes `setup_kwargs`. If they do, env raises `TypeError` — correct loud failure. |
| Dashboard render of v1 vs v2 manifests | Briefly broken between deploying v2 manifests and shipping the JS update. Acceptable. |