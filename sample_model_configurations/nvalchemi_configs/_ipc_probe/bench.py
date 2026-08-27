"""IPC probe driver: correctness, NVE parity, and overhead grid.

Runs in the main venv (nvalchemi engine + rootstock client). The
baseline runs the same workloads in-process inside the worker venv via
baseline.py, from the same systems .npz, so every comparison is
like-for-like.

    python bench.py --family mace --root /rs-root --env mace \\
        --checkpoint mace-medium-0b2-batched --device cuda --mode all
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

FAMILY_KIND = {"mace": "periodic", "uma": "molecular", "aimnet2": "molecular"}
FAMILY_CHARGE = {"mace": False, "uma": True, "aimnet2": True}
FAMILY_SPIN = {"mace": False, "uma": True, "aimnet2": False}


def stage(msg):
    print(f"STAGE: {msg}", flush=True)


def run_baseline(args, systems_npz, workdir, *, forward_iters=0, nve_steps=0, nve_dt=1.0):
    from pathlib import Path

    from rootstock.environment import get_model_cache_env

    job = {
        "env_dir": os.path.join(args.root, "envs", args.env),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "setup_kwargs": json.loads(args.setup_kwargs),
        "systems_npz": systems_npz,
        "with_charge": FAMILY_CHARGE[args.family],
        "with_spin": FAMILY_SPIN[args.family],
        "forward_iters": forward_iters,
        "nve_steps": nve_steps,
        "nve_dt": nve_dt,
        "out_npz": os.path.join(workdir, "baseline_out.npz"),
    }
    job_path = os.path.join(workdir, "job.json")
    with open(job_path, "w") as f:
        json.dump(job, f)

    env = os.environ.copy()
    env.update(get_model_cache_env(Path(args.root)))
    python = os.path.join(args.root, "envs", args.env, "bin", "python")
    proc = subprocess.run(
        [python, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.py"), job_path],
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"baseline failed (rc={proc.returncode})")
    timings = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON: "):
            timings = json.loads(line[len("RESULT_JSON: ") :])
    import numpy as np

    return np.load(job["out_npz"]), timings


def make_model(args):
    from rootstock.batched.model import AlchemiModel

    return AlchemiModel(
        args.checkpoint,
        root=args.root,
        device=args.device,
        setup_kwargs=json.loads(args.setup_kwargs),
        neighbor_mode=args.neighbor_mode,
        transport=args.transport,
    )


def proxy_forward(model, batch):
    """One proxied forward, with the engine-side NL hook when in engine mode."""
    if model.neighbor_mode == "engine":
        from nvalchemi.neighbors import compute_neighbors

        compute_neighbors(batch, config=model.model_config.neighbor_config)
    return model(batch)


def correctness(args, model, workdir, report):
    import numpy as np

    stage("correctness: single forward, proxy vs in-process")
    systems = common.make_systems(FAMILY_KIND[args.family], 4, args.correctness_atoms)
    systems_npz = os.path.join(workdir, "systems.npz")
    common.save_systems(systems_npz, systems)
    baseline_out, _ = run_baseline(args, systems_npz, workdir)

    batch = common.build_batch(
        systems,
        args.device,
        with_charge=FAMILY_CHARGE[args.family],
        with_spin=FAMILY_SPIN[args.family],
    )
    outputs = proxy_forward(model, batch)
    diffs = {}
    for key in ("energy", "forces", "stress"):
        value = outputs.get(key)
        if value is not None and key in baseline_out:
            delta = np.abs(value.detach().cpu().double().numpy() - baseline_out[key])
            ref = np.abs(baseline_out[key]).max() or 1.0
            diffs[key] = {"max_abs": float(delta.max()), "max_rel": float(delta.max() / ref)}
    report["correctness"] = diffs
    print(json.dumps(diffs, indent=2), flush=True)


def nve_parity(args, model, workdir, report):
    import numpy as np

    stage(f"nve parity: {args.nve_steps} steps, proxy engine vs in-process engine")
    systems = common.make_systems(FAMILY_KIND[args.family], 4, args.correctness_atoms)
    systems_npz = os.path.join(workdir, "systems_nve.npz")
    common.save_systems(systems_npz, systems)
    baseline_out, baseline_t = run_baseline(
        args, systems_npz, workdir, nve_steps=args.nve_steps, nve_dt=args.nve_dt
    )

    batch = common.build_batch(
        systems,
        args.device,
        with_charge=FAMILY_CHARGE[args.family],
        with_spin=FAMILY_SPIN[args.family],
    )
    t0 = time.perf_counter()
    positions, _ = common.run_nve(
        model,
        batch,
        steps=args.nve_steps,
        dt=args.nve_dt,
        register_engine_nl=(args.neighbor_mode == "engine"),
    )
    proxy_s = time.perf_counter() - t0
    delta = float(np.abs(positions - baseline_out["nve_positions"]).max())
    result = {
        "max_abs_position_diff": delta,
        "proxy_nve_s": proxy_s,
        "baseline_nve_s": baseline_t.get("nve_s"),
    }
    report["nve"] = result
    print(json.dumps(result, indent=2), flush=True)


def perf_grid(args, model, workdir, report):
    stage("perf grid")
    grid = []
    for spec in args.grid.split(","):
        n_sys, n_atoms = spec.strip().split("x")
        grid.append((int(n_sys), int(n_atoms)))

    rows = []
    for n_sys, n_atoms in grid:
        systems = common.make_systems(FAMILY_KIND[args.family], n_sys, n_atoms)
        systems_npz = os.path.join(workdir, f"systems_{n_sys}x{n_atoms}.npz")
        common.save_systems(systems_npz, systems)
        model.clear_worker_cache()
        baseline_out, baseline_t = run_baseline(
            args, systems_npz, workdir, forward_iters=args.iters
        )
        del baseline_out

        batch = common.build_batch(
            systems,
            args.device,
            with_charge=FAMILY_CHARGE[args.family],
            with_spin=FAMILY_SPIN[args.family],
        )
        for _ in range(3):
            proxy_forward(model, batch)
        model.stats.clear()
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            proxy_forward(model, batch)
            samples.append(time.perf_counter() - t0)

        base_med = statistics.median(baseline_t["forward_s"])
        proxy_med = statistics.median(samples)
        last = model.stats[-1]
        row = {
            "n_systems": n_sys,
            "atoms_per_system": n_atoms,
            "baseline_median_s": base_med,
            "proxy_median_s": proxy_med,
            "overhead_pct": 100.0 * (proxy_med - base_med) / base_med,
            "bytes_sent": last["bytes_sent"],
            "bytes_received": last["bytes_received"],
            "decomposition_last": {
                "gather_s": last["gather_s"],
                "roundtrip_s": last["roundtrip_s"],
                "to_device_s": last["to_device_s"],
                "worker": last["worker"],
            },
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report["perf"] = rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_KIND))
    parser.add_argument("--root", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="all", choices=["correctness", "nve", "perf", "all"])
    parser.add_argument("--neighbor-mode", default="worker", choices=["worker", "engine"])
    parser.add_argument("--transport", default="socket", choices=["socket", "cuda"])
    parser.add_argument("--setup-kwargs", default="{}")
    parser.add_argument("--correctness-atoms", type=int, default=96)
    parser.add_argument("--nve-steps", type=int, default=100)
    parser.add_argument("--nve-dt", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--grid", default="1x64,8x64,32x64,8x512")
    args = parser.parse_args()

    workdir = tempfile.mkdtemp(prefix="ipc_probe_")
    report = {
        "family": args.family,
        "neighbor_mode": args.neighbor_mode,
        "transport": args.transport,
        "device": args.device,
    }

    stage(f"starting proxy worker ({args.family}, neighbor_mode={args.neighbor_mode})")
    model = make_model(args)
    report["worker_info"] = {
        k: v for k, v in model._worker_info.items() if k not in ("model_config", "tensors")
    }
    try:
        if args.mode in ("correctness", "all"):
            correctness(args, model, workdir, report)
        if args.mode in ("nve", "all"):
            nve_parity(args, model, workdir, report)
        if args.mode in ("perf", "all"):
            perf_grid(args, model, workdir, report)
    finally:
        model.close()

    stage("done")
    print("REPORT_JSON: " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
