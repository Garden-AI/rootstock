"""In-process baseline: runs the real wrapper in the worker venv.

Invoked by bench.py with the env's Python:

    envs/<fam>/bin/python baseline.py <job.json>

The job file names the env dir (for env_source import), the systems
.npz, and what to run. Results land in an .npz next to the job file;
timings print as a final ``RESULT_JSON: {...}`` line.
"""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    with open(sys.argv[1]) as f:
        job = json.load(f)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, job["env_dir"])
    import common
    import numpy as np
    import torch
    from env_source import setup_batched

    result = setup_batched(job["checkpoint"], job["device"], **job.get("setup_kwargs", {}))
    wrapper, options = result if isinstance(result, tuple) else (result, {})
    cfg = wrapper.model_config
    compute_nl = options.get("compute_neighbors", cfg.needs_neighborlist)

    systems = common.load_systems(job["systems_npz"])
    batch = common.build_batch(
        systems,
        job["device"],
        with_charge=job.get("with_charge", False),
        with_spin=job.get("with_spin", False),
    )
    device = torch.device(job["device"])

    def one_forward():
        if compute_nl:
            from nvalchemi.neighbors import compute_neighbors

            compute_neighbors(batch, config=cfg.neighbor_config)
        outputs = wrapper(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return outputs

    out: dict[str, np.ndarray] = {}
    timings: dict[str, object] = {}

    outputs = one_forward()
    for key in ("energy", "forces", "stress"):
        value = outputs.get(key)
        if value is not None:
            out[key] = value.detach().cpu().double().numpy()

    iters = job.get("forward_iters", 0)
    if iters:
        for _ in range(3):
            one_forward()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            one_forward()
            samples.append(time.perf_counter() - t0)
        timings["forward_s"] = samples

    steps = job.get("nve_steps", 0)
    if steps:
        # Fresh batch: the timing loop above moved nothing, but keep NVE
        # inputs identical to the proxy run, which also starts clean.
        batch = common.build_batch(
            systems,
            job["device"],
            with_charge=job.get("with_charge", False),
            with_spin=job.get("with_spin", False),
        )
        t0 = time.perf_counter()
        positions, velocities = common.run_nve(
            wrapper,
            batch,
            steps=steps,
            dt=job.get("nve_dt", 1.0),
            register_engine_nl=compute_nl,
        )
        timings["nve_s"] = time.perf_counter() - t0
        out["nve_positions"] = positions.astype(np.float64)
        out["nve_velocities"] = velocities.astype(np.float64)

    np.savez(job["out_npz"], **out)
    print("RESULT_JSON: " + json.dumps(timings), flush=True)


if __name__ == "__main__":
    main()
