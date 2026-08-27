"""Diagnose the UMA in-worker forward slowdown.

Runs standalone in the worker venv (no IPC). Times wrapper(batch) under
input regimes that differ exactly the way the batched worker differs from
the in-process baseline, plus a contention regime with a sibling process
holding an idle CUDA context (as the engine process does in production).

    envs/uma/bin/python uma_diag.py <env_dir> <device>

Regimes:
    R1_same_batch       same Batch object every call (baseline behavior)
    R2_new_ptr          same Batch, positions replaced by a fresh clone each
                        call (new data_ptr, identical values)
    R2b_h2d_setattr     fresh CPU->GPU tensors assigned into the Batch each
                        call (exact batched-worker steady-state path)
    R3_rebuild          Batch rebuilt from AtomicData list each call
                        (batched-worker resegmentation path)
    R4_idle_context     R1 again while a sibling process holds an idle CUDA
                        context on the same GPU
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

WARMUP = 5
ITERS = 20


def timed(fn, device):
    import torch

    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    env_dir, device_str = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, env_dir)
    import common
    import torch
    from env_source import setup_batched

    device = torch.device(device_str)
    result = setup_batched("uma-s-1p1-batched", device_str, task="omol")
    wrapper, _ = result if isinstance(result, tuple) else (result, {})
    print(
        "inference_settings:",
        getattr(wrapper.predict_unit, "inference_settings", "<none>"),
        flush=True,
    )

    systems = common.make_systems("molecular", 1, 64)
    report = {}

    def fresh_batch():
        return common.build_batch(systems, device_str, with_charge=True, with_spin=True)

    # R1: identical call pattern to the in-process baseline.
    batch = fresh_batch()
    report["R1_same_batch"] = timed(lambda: wrapper(batch), device)

    # R2: same batch object, but positions gets a new GPU allocation each
    # call (identical values) — isolates data_ptr churn.
    batch2 = fresh_batch()

    def r2():
        batch2.positions = batch2.positions.clone()
        wrapper(batch2)

    report["R2_new_ptr"] = timed(r2, device)

    # R2b: the batched worker's steady-state path — fresh host tensors,
    # H2D copy, setattr into the existing Batch.
    batch3 = fresh_batch()
    host = {
        "positions": batch3.positions.cpu().clone(),
        "atomic_numbers": batch3.atomic_numbers.cpu().clone(),
        "cell": batch3.cell.cpu().clone(),
        "pbc": batch3.pbc.cpu().clone(),
        "charge": batch3.charge.cpu().clone(),
        "spin": batch3.spin.cpu().clone(),
    }

    def r2b():
        for key, value in host.items():
            setattr(batch3, key, value.to(device))
        wrapper(batch3)

    report["R2b_h2d_setattr"] = timed(r2b, device)

    # R3: full rebuild each call.
    def r3():
        wrapper(fresh_batch())

    report["R3_rebuild"] = timed(r3, device)

    # R5: shape churn — serve many distinct batch shapes (as the probe
    # worker did across correctness/NVE/grid), then re-time the 1x64 shape.
    # If torch._dynamo hits its recompile limit it falls back to eager for
    # good, which would explain the probe's 2.3x in-worker slowdown.
    for n_sys, n_atoms in [(4, 96), (2, 48), (3, 80), (8, 64), (5, 33), (6, 129), (2, 200)]:
        churn = common.build_batch(
            common.make_systems("molecular", n_sys, n_atoms),
            device_str,
            with_charge=True,
            with_spin=True,
        )
        wrapper(churn)
    batch5 = fresh_batch()
    report["R5_after_shape_churn"] = timed(lambda: wrapper(batch5), device)
    try:
        import torch._dynamo as dynamo

        report["dynamo_cache_limit"] = dynamo.config.cache_size_limit
        counters = dynamo.utils.counters
        report["dynamo_counters"] = {
            k: dict(v) for k, v in counters.items() if k in ("stats", "frames")
        }
    except Exception:
        pass

    # R6: per-call active_outputs reassignment — the batched worker does
    # this on every compute; none of the regimes above do.
    batch6 = fresh_batch()

    def r6():
        wrapper.model_config.active_outputs = {"energy", "forces"}
        wrapper(batch6)

    report["R6_active_reassign"] = timed(r6, device)

    # R7: ground truth — the exact batched-worker compute path, in-process:
    # numpy host tensors -> H2D -> _BatchCache.update -> active reassignment
    # -> forward. If this stays fast, the slowdown needs the real two-process
    # setup and is environmental, not in the worker code.
    from rootstock.batched.worker import _BatchCache

    ref = fresh_batch()
    arrays = {
        "positions": ref.positions.cpu().numpy(),
        "atomic_numbers": ref.atomic_numbers.cpu().numpy(),
        "num_nodes_per_graph": ref.num_nodes_per_graph.cpu().numpy(),
        "cell": ref.cell.cpu().numpy(),
        "pbc": ref.pbc.cpu().numpy(),
        "charge": ref.charge.cpu().numpy(),
        "spin": ref.spin.cpu().numpy(),
    }
    cache = _BatchCache(device)

    def r7():
        import numpy as np

        tensors = {k: torch.from_numpy(np.array(v)).to(device) for k, v in arrays.items()}
        b = cache.update(tensors)
        wrapper.model_config.active_outputs = {"energy", "forces"}
        wrapper(b)

    report["R7_worker_path"] = timed(r7, device)
    report["lazy_model_intialized"] = getattr(
        wrapper.predict_unit, "lazy_model_intialized", "<missing>"
    )

    # R8: MD-style drift — 100 forwards with perturbed positions (edge count
    # changes call to call, as it did while the probe worker served the NVE
    # stage), then re-time. If dynamo's per-frame recompile limit (32) trips
    # on the edge-dynamic frame, fairchem's compiled path falls back to eager
    # permanently — the probe's perf grid ran after exactly such a stage.
    batch8 = fresh_batch()
    base_pos = batch8.positions.clone()
    for step in range(100):
        batch8.positions = base_pos + torch.randn_like(base_pos) * (0.01 * (step + 1))
        wrapper(batch8)
    batch8.positions = base_pos
    report["R8_after_md_drift"] = timed(lambda: wrapper(batch8), device)

    # R9: after the drift exhausted the recompile limit, present shapes the
    # process has NEVER compiled. If dynamo now refuses to compile new
    # variants, these run eager — the probe's perf grid did exactly this.
    fresh_a = common.build_batch(
        common.make_systems("molecular", 2, 64), device_str, with_charge=True, with_spin=True
    )
    report["R9_new_shape_2x64"] = timed(lambda: wrapper(fresh_a), device)
    fresh_b = common.build_batch(
        common.make_systems("molecular", 1, 63), device_str, with_charge=True, with_spin=True
    )
    report["R9_new_shape_1x63"] = timed(lambda: wrapper(fresh_b), device)
    try:
        import torch._dynamo as dynamo

        report["dynamo_counters_after_drift"] = {
            k: dict(v) for k, v in dynamo.utils.counters.items()
        }
    except Exception:
        pass

    # R4: R1 while a sibling process holds an idle CUDA context.
    sibling = None
    if device.type == "cuda":
        sibling = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import torch; torch.zeros(1, device='cuda'); import time; time.sleep(600)",
            ]
        )
        time.sleep(15)  # let the context come up
        batch4 = fresh_batch()
        report["R4_idle_context"] = timed(lambda: wrapper(batch4), device)
        sibling.terminate()

    print("DIAG_JSON: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
