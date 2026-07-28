# /// script
# requires-python = ">=3.11"
# dependencies = ["rootstock", "ase>=3.22", "numpy"]
#
# [tool.uv]
# python-preference = "managed"
# ///
"""Measure Rootstock's i-PI IPC overhead: managed calculator vs. in-env direct.

The question this answers: when you run an MLIP through Rootstock's
``RootstockCalculator`` (which talks to a worker subprocess over a Unix socket
using the i-PI protocol), how much slower is each force evaluation than calling
the *same* calculator directly, with no IPC in the loop?

Methodology
-----------
Both arms load the calculator via the pre-built env's
``env_source.setup(checkpoint, device, **setup_kwargs)`` -- the identical code
the real worker runs -- so the only difference is the transport:

  * rootstock arm  -- RootstockCalculator: every force call crosses the socket.
  * in-env arm     -- the env's own python imports setup() and calls the
                      calculator directly, no socket.

The in-env arm is run by re-spawning this file under the env's interpreter
(``{root}/envs/{env}/bin/python this_file.py --worker-mode``) -- the same
mechanism Rootstock uses to spawn workers. The worker-mode path only needs
``ase`` + ``numpy``, both present in every env.

Both arms replay the *identical* sequence of atomic positions (serialized once,
fed to both), so per-call compute is the same configuration-for-configuration
and the timing delta is purely the IPC hop. This is the inner loop of MD and
relaxation, so it is indicative of a real workload.

What it reports, per (checkpoint, device)
-----------------------------------------
  * steady-state per-call latency (median / mean / p95) for each arm
  * IPC overhead: rootstock_median - in_env_median, absolute and as a %
  * startup cost (model load + worker spawn) -- a one-time, per-session cost

Usage
-----
With the ``rootstock`` CLI installed (no repo checkout needed) -- list what is
installed on this cluster, then pick a few checkpoints::

    rootstock benchmark --root /projects/bchg/rootstock --list

    rootstock benchmark \
        --root /projects/bchg/rootstock \
        --checkpoints mace-mp-0-medium uma-s-1p1 sevennet-0 \
        --devices cuda \
        --system Cu:256 \
        --calls 100 --warmup 10 \
        --out bench.json

``--root`` (or ``--cluster`` for a registered cluster) is the only
cluster-specific knob; everything else is identical across machines. Add
``--devices cuda cpu`` to cover both. Run inside a GPU allocation for cuda.

From a repo checkout the same entry point is also reachable as
``uv run rootstock/benchmark.py ...`` (the PEP 723 header above lets uv resolve
deps on the fly) or ``python -m rootstock.benchmark ...``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# System construction (shared by both arms; deterministic).
# ---------------------------------------------------------------------------


def make_atoms(system: str, seed: int = 42):
    """Build the benchmark system from a compact spec.

    Specs:
      ``Cu:256``    -- FCC bulk supercell of element Cu, ~256 atoms (periodic).
                       Any ASE-known bulk element works (``Si:512``, ``Fe:128``).
      ``H2O``       -- an ASE ``molecule`` (non-periodic), centered in vacuum.

    Cu is the default because it is a simple FCC metal that every *universal*
    MLIP supports and scales smoothly to large supercells. Use a molecule spec
    for organic-only models (e.g. ANI).
    """
    from ase.build import bulk, molecule

    if ":" in system:
        element, count = system.split(":")
        target = int(count)
        # FCC conventional cell has 4 atoms; pick n so 4*n^3 ~= target.
        n = max(1, round((target / 4) ** (1 / 3)))
        atoms = bulk(element, "fcc", a=3.615, cubic=True) * (n, n, n)
    else:
        atoms = molecule(system)
        atoms.center(vacuum=5.0)

    return atoms


def make_trajectory(atoms, n_frames: int, rattle: float, seed: int) -> np.ndarray:
    """Return an ``(n_frames, n_atoms, 3)`` array of perturbed positions.

    A seeded random walk away from the base geometry: small Gaussian steps that
    keep the neighbour list essentially fixed (so per-call compute is constant)
    while still moving the atoms, mimicking the inner loop of MD/relaxation.
    Both arms replay this exact array.
    """
    rng = np.random.default_rng(seed)
    base = atoms.get_positions()
    frames = np.empty((n_frames, *base.shape), dtype=np.float64)
    pos = base.copy()
    for i in range(n_frames):
        pos = pos + rng.normal(0.0, rattle, base.shape)
        frames[i] = pos
    return frames


# ---------------------------------------------------------------------------
# Timing primitives.
# ---------------------------------------------------------------------------


def _summarize(times_ms: list[float]) -> dict:
    a = np.asarray(times_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "std_ms": float(np.std(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(np.min(a)),
        "max_ms": float(np.max(a)),
        "n": int(a.size),
    }


def time_force_loop(atoms, calc, frames: np.ndarray, n_warmup: int) -> dict:
    """Replay ``frames`` through ``calc``, returning timing + startup info.

    The first ``n_warmup`` frames are timed-but-discarded (JIT, GPU alloc,
    neighbour-list build). The very first call's wall time is also reported
    separately as ``startup_s`` -- it folds in model warmup and, for the
    rootstock arm, the first round-trip handshake.
    """
    atoms = atoms.copy()
    atoms.calc = calc

    times_ms: list[float] = []
    startup_s = None
    for i, pos in enumerate(frames):
        atoms.set_positions(pos)
        t0 = time.perf_counter()
        atoms.get_forces()
        dt = time.perf_counter() - t0
        if i == 0:
            startup_s = dt
        if i >= n_warmup:
            times_ms.append(dt * 1000.0)

    summary = _summarize(times_ms)
    summary["startup_s"] = startup_s
    return summary


# ---------------------------------------------------------------------------
# In-env worker mode: runs *inside* the pre-built env's python.
# ---------------------------------------------------------------------------


def run_worker_mode(args) -> int:
    """Entry point executed by ``{root}/envs/{env}/bin/python``.

    Loads the env's calculator via ``env_source.setup()`` -- the same function
    Rootstock's real worker calls -- and times a direct force loop with no IPC.
    Emits one JSON line to stdout: ``RESULT <json>``.
    """
    sys.path.insert(0, args.env_dir)

    data = np.load(args.worker_data)
    from ase import Atoms

    atoms = Atoms(
        numbers=data["numbers"],
        positions=data["frames"][0],
        cell=data["cell"],
        pbc=data["pbc"],
    )
    if "charge" in data:
        atoms.info["charge"] = int(data["charge"])
    if "spin" in data:
        atoms.info["spin"] = int(data["spin"])

    setup_kwargs = json.loads(args.setup_kwargs) if args.setup_kwargs else {}

    # Mirror the real worker wrapper's branch: local checkpoints load through
    # the env's setup_from_path hook, canonical ids through setup().
    t0 = time.perf_counter()
    if getattr(args, "checkpoint_path", None):
        from env_source import setup_from_path  # type: ignore

        calc = setup_from_path(args.checkpoint_path, args.device, **setup_kwargs)
    else:
        from env_source import setup  # type: ignore

        calc = setup(args.checkpoint, args.device, **setup_kwargs)
    load_s = time.perf_counter() - t0

    result = time_force_loop(atoms, calc, data["frames"], int(data["n_warmup"]))
    result["model_load_s"] = load_s
    print("RESULT " + json.dumps(result), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Driver: orchestrates both arms for each (checkpoint, device).
# ---------------------------------------------------------------------------


def run_in_env_arm(root: Path, cache_root: Path | None, env_name: str, env_dir: Path,
                   checkpoint: str, device: str, setup_kwargs: dict,
                   npz_path: Path, checkpoint_path: str | None = None) -> dict:
    """Spawn the env's python in worker mode and parse its JSON result.

    Reproduces Rootstock's own subprocess environment (the HOME/cache redirect
    from ``get_model_cache_env``) so the in-env arm finds the identical cached
    weights the managed arm would -- otherwise the two arms would differ on
    cache state, not just transport.
    """
    from rootstock.environment import get_model_cache_env

    env_python = env_dir / "bin" / "python"
    if not env_python.exists():
        raise RuntimeError(f"env python not found: {env_python}")

    env = os.environ.copy()
    env.update(get_model_cache_env(root, cache_root))

    cmd = [
        str(env_python),
        os.path.abspath(__file__),
        "--worker-mode",
        "--env-dir", str(env_dir),
        "--checkpoint", checkpoint,
        "--device", device,
        "--worker-data", str(npz_path),
        "--setup-kwargs", json.dumps(setup_kwargs),
    ]
    if checkpoint_path:
        cmd += ["--checkpoint-path", checkpoint_path]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"in-env worker failed (code {proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"no RESULT line from worker.\nstdout:\n{proc.stdout}")


def run_rootstock_arm(root: Path, cache_root: Path | None, cluster: str | None,
                      checkpoint: str, device: str, setup_kwargs: dict,
                      atoms, frames: np.ndarray, n_warmup: int,
                      weights: str | None = None) -> dict:
    """Time the same force loop through RootstockCalculator (IPC in the loop)."""
    from rootstock import RootstockCalculator

    kwargs: dict = {"checkpoint": checkpoint, "device": device,
                    "setup_kwargs": setup_kwargs}
    if weights:
        kwargs["weights"] = weights
    if cluster:
        kwargs["cluster"] = cluster
    else:
        kwargs["root"] = str(root)
    if cache_root is not None:
        kwargs["cache_root"] = str(cache_root)

    with RootstockCalculator(**kwargs) as calc:
        return time_force_loop(atoms, calc, frames, n_warmup)


def benchmark_one(checkpoint: str, device: str, root: Path, cache_root: Path | None,
                  cluster: str | None, atoms, frames: np.ndarray, n_warmup: int,
                  setup_kwargs: dict, work_dir: Path,
                  weights: str | None = None) -> dict:
    from rootstock.environment import bind_custom_weights
    from rootstock.local_checkpoints import resolve_checkpoint

    resolved = resolve_checkpoint(root, checkpoint)
    # Same guards as the calculator — fail before either arm spawns, not as
    # a raw subprocess traceback from the in-env worker.
    custom_path = bind_custom_weights(root, resolved.env_name, checkpoint, weights, setup_kwargs)
    checkpoint_path = custom_path if custom_path is not None else resolved.path
    if resolved.is_local and not Path(resolved.path).exists():
        raise RuntimeError(
            f"local checkpoint '{checkpoint}' points at {resolved.path}, "
            f"which no longer exists. Re-register it with `rootstock "
            f"add-local` or remove it with `rootstock remove-local "
            f"{checkpoint}`."
        )
    env_name = resolved.env_name
    env_dir = root / "envs" / env_name
    # Registered defaults for a local checkpoint; explicit --setup-kwargs
    # wins. Both arms get the identical merged dict (the managed arm's
    # calculator re-merges, with the same precedence, to the same result).
    setup_kwargs = {**resolved.setup_kwargs, **setup_kwargs}

    # Serialize the system + identical trajectory once; both arms read it.
    npz_path = work_dir / f"frames_{checkpoint.replace('/', '_')}_{device}.npz"
    np.savez(
        npz_path,
        numbers=atoms.get_atomic_numbers(),
        cell=np.array(atoms.cell),
        pbc=np.array(atoms.pbc),
        frames=frames,
        n_warmup=np.array(n_warmup),
    )

    print(f"  [in-env]    {checkpoint} on {device} via {env_name} ...", flush=True)
    direct = run_in_env_arm(root, cache_root, env_name, env_dir, checkpoint,
                            device, setup_kwargs, npz_path,
                            checkpoint_path=checkpoint_path)

    print(f"  [rootstock] {checkpoint} on {device} (IPC) ...", flush=True)
    rs = run_rootstock_arm(root, cache_root, cluster, checkpoint, device,
                           setup_kwargs, atoms, frames, n_warmup,
                           weights=weights)

    overhead_ms = rs["median_ms"] - direct["median_ms"]
    overhead_pct = (
        100.0 * overhead_ms / direct["median_ms"] if direct["median_ms"] else float("nan")
    )

    return {
        "checkpoint": checkpoint,
        "env": env_name,
        "device": device,
        "n_atoms": int(len(atoms)),
        "in_env": direct,
        "rootstock": rs,
        "overhead_ms_median": overhead_ms,
        "overhead_pct_median": overhead_pct,
        "startup_overhead_s": (rs.get("startup_s") or 0) - (direct.get("startup_s") or 0),
    }


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 92)
    print("IPC overhead: RootstockCalculator vs. in-environment direct call")
    print("=" * 92)
    header = (f"{'checkpoint':<24}{'dev':<5}{'atoms':>6}  "
              f"{'in-env ms':>10}{'rootstock ms':>13}{'overhead ms':>12}{'overhead %':>11}")
    print(header)
    print("-" * 92)
    for r in results:
        if "error" in r:
            print(f"{r['checkpoint']:<24}{r['device']:<5}{'':>6}  ERROR: {r['error'][:48]}")
            continue
        print(f"{r['checkpoint']:<24}{r['device']:<5}{r['n_atoms']:>6}  "
              f"{r['in_env']['median_ms']:>10.2f}{r['rootstock']['median_ms']:>13.2f}"
              f"{r['overhead_ms_median']:>12.2f}{r['overhead_pct_median']:>10.1f}%")
    print("-" * 92)
    print("median per-call force-eval latency; overhead = rootstock - in-env.")
    print("startup (model load + worker spawn) is reported per-checkpoint in the JSON.\n")


def list_available(root: Path) -> int:
    from rootstock.environment import list_declared_checkpoints
    from rootstock.local_checkpoints import LocalCheckpointError, local_checkpoints_for_root

    declared = list_declared_checkpoints(root)
    if not declared:
        print(f"No envs installed at {root}. Run `rootstock install` first.")
        return 1
    print(f"Checkpoints declared by installed envs at {root}:\n")
    for env, ckpts in declared.items():
        ids = ", ".join(ckpts) if ckpts else "(none)"
        print(f"  {env:<16} {ids}")
    try:
        local = local_checkpoints_for_root(root)
    except LocalCheckpointError as exc:
        print(f"Warning: ignoring local-checkpoint registry: {exc}", file=sys.stderr)
        local = {}
    if local:
        ids = ", ".join(sorted(local))
        print(f"  {'(local)':<16} {ids}")
    print("\nPass a few of these to --checkpoints.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="rootstock benchmark",
        description="Benchmark Rootstock i-PI IPC overhead vs. in-env direct calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Cluster location (one of cluster / root).
    p.add_argument("--root", help="Rootstock install root (e.g. /projects/bchg/rootstock).")
    p.add_argument("--cluster", help="Registered cluster name (alternative to --root).")
    p.add_argument("--cache-root", help="Separate weight-cache root, if split from --root.")

    p.add_argument("--checkpoints", nargs="+", help="Canonical checkpoint ids to benchmark.")
    p.add_argument("--devices", nargs="+", default=["cuda"], help="Devices (cuda, cpu).")
    p.add_argument("--system", default="Cu:256",
                   help="System spec: 'El:N' bulk supercell, or an ASE molecule name.")
    p.add_argument("--calls", type=int, default=100, help="Timed force calls per arm.")
    p.add_argument("--warmup", type=int, default=10, help="Untimed warmup calls.")
    p.add_argument("--rattle", type=float, default=0.005,
                   help="Per-step Gaussian displacement (A) for the replay trajectory.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--setup-kwargs", default="",
                   help='JSON forwarded to setup() for every checkpoint '
                        '(e.g. \'{"task":"omat"}\').')
    p.add_argument("--weights",
                   help="Path to your own weights file; requires a '<family>:custom' "
                        "entry in --checkpoints (applies to every checkpoint given).")
    p.add_argument("--out", help="Write full results JSON here.")
    p.add_argument("--list", action="store_true", help="List installed checkpoints and exit.")

    # Hidden worker-mode flags (used when this file is re-spawned by env python).
    p.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--env-dir", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint-path", help=argparse.SUPPRESS)
    p.add_argument("--device", help=argparse.SUPPRESS)
    p.add_argument("--worker-data", help=argparse.SUPPRESS)

    args = p.parse_args(argv)

    if args.worker_mode:
        return run_worker_mode(args)

    if not (args.root or args.cluster):
        p.error("one of --root or --cluster is required")

    # Resolve root and cache_root (needed even with --cluster: the in-env arm
    # builds its cache env from these paths directly, never going through
    # RootstockCalculator's cluster resolution — so a split cache_root must be
    # threaded through here or that arm looks for weights under `root`).
    # cache_root resolves through resolve_cache_root like every other entry
    # point, so a split cache declared in {root}/layout.json (e.g. Frontier:
    # root on read-only /sw, weights on Lustre) is honored with a bare --root,
    # not only via --cache-root or the cluster registry.
    cluster_info = None
    if args.cluster:
        from rootstock.clusters import get_cluster
        cluster_info = get_cluster(args.cluster)
    root = Path(args.root) if args.root else Path(cluster_info.root)
    from rootstock.layout import resolve_cache_root
    cache_root = resolve_cache_root(root, explicit=args.cache_root)

    if args.list:
        return list_available(root)

    if not args.checkpoints:
        p.error("--checkpoints is required (run with --list to see what's installed)")

    setup_kwargs = json.loads(args.setup_kwargs) if args.setup_kwargs else {}

    atoms = make_atoms(args.system, seed=args.seed)
    n_frames = args.calls + args.warmup
    frames = make_trajectory(atoms, n_frames, args.rattle, args.seed)

    print(f"System: {args.system} -> {len(atoms)} atoms; "
          f"{args.calls} timed calls (+{args.warmup} warmup); "
          f"root={root}")

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="rootstock_bench_") as tmp:
        work_dir = Path(tmp)
        for device in args.devices:
            for checkpoint in args.checkpoints:
                print(f"\n>>> {checkpoint} @ {device}")
                try:
                    r = benchmark_one(checkpoint, device, root, cache_root, args.cluster,
                                      atoms, frames, args.warmup, setup_kwargs, work_dir,
                                      weights=args.weights)
                    print(f"    overhead: {r['overhead_ms_median']:.2f} ms/call "
                          f"({r['overhead_pct_median']:.1f}%)")
                except Exception as e:  # noqa: BLE001 - one bad model shouldn't abort the rest
                    import traceback
                    traceback.print_exc()
                    r = {"checkpoint": checkpoint, "device": device, "error": str(e)}
                results.append(r)

    print_table(results)

    payload = {
        "system": args.system,
        "n_calls": args.calls,
        "n_warmup": args.warmup,
        "rattle": args.rattle,
        "seed": args.seed,
        "root": str(root),
        "setup_kwargs": setup_kwargs,
        "results": results,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
