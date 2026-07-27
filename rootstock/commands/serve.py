"""Serve command for starting worker processes."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

from .common import get_root_or_exit


def cmd_serve(args) -> int:
    """
    Start a rootstock worker process for an external i-PI server (e.g., LAMMPS).

    The worker connects to the given Unix socket path, loads the specified model,
    and serves energy/forces via the i-PI protocol until the server disconnects.

    Exit codes:
        0: Clean shutdown
        1: Error
    """
    from ..environment import (
        CheckpointNotFoundError,
        CustomWeightsError,
        bind_custom_weights,
        get_env_python,
        resolve_checkpoint,
    )
    from ..manifest import now_iso
    from ..operations import parse_setup_kwargs
    from ..spawn import WORKER_WRAPPER, spawn_in_env
    from ..usage import record_session
    from .common import resolve_cache_root

    root = get_root_or_exit(args)
    cache_root = resolve_cache_root(root)
    socket_path = args.socket
    checkpoint = args.checkpoint
    device = args.device
    weights = getattr(args, "weights", None)

    try:
        cli_kwargs = parse_setup_kwargs(getattr(args, "kwarg", None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        resolved = resolve_checkpoint(root, checkpoint)
    except CheckpointNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    env_name = resolved.env_name
    setup_kwargs = cli_kwargs

    # Enforce the ':custom' / --weights pairing (both directions) and, for a
    # custom id, validate the file + the built env's setup_from_path hook —
    # before the startup banner, not as an ImportError inside the worker.
    # None for a canonical id.
    try:
        checkpoint_path = bind_custom_weights(root, env_name, checkpoint, weights, cli_kwargs)
    except CustomWeightsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Validate environment exists before printing the startup banner
    try:
        get_env_python(root, env_name)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Starting rootstock worker:")
    print(f"  Env: {env_name}")
    if resolved.is_custom:
        print(f"  Checkpoint: {checkpoint} (weights: {checkpoint_path})")
    else:
        print(f"  Checkpoint: {checkpoint}")
    print(f"  Device: {device}")
    print(f"  Socket: {socket_path}")

    # The context must stay open until the worker exits — it reads the
    # wrapper and sidecar at startup.
    with spawn_in_env(
        root,
        env_name,
        WORKER_WRAPPER,
        {
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "socket_path": socket_path,
            "setup_kwargs": setup_kwargs,
        },
        cache_root=cache_root,
    ) as spec:
        started_at = now_iso()
        started = time.monotonic()
        proc = subprocess.Popen(spec.cmd, env=spec.env, cwd=spec.cwd)

        # Forward signals to worker
        def forward_signal(signum, frame):
            proc.send_signal(signum)

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)

        # Block until worker exits
        rc = proc.wait()

    # One anonymous usage record per serve session (see usage.py) — the only
    # entry point not covered by the RootstockServer hook, because here the
    # i-PI server is external (e.g. the LAMMPS fix). This parent survives a
    # walltime SIGTERM (the handler above only forwards it), so killed runs
    # still record. The worker's force-call count isn't visible from here:
    # n_calculations is null; sessions and wall-time still count.
    record_session(
        root=root,
        cache_root=cache_root,
        env_name=env_name,
        checkpoint=checkpoint,
        device=device,
        client="serve",
        started_at=started_at,
        duration_s=time.monotonic() - started,
        n_calculations=None,
    )
    return rc
