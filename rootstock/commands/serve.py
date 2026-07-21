"""Serve command for starting worker processes."""

from __future__ import annotations

import signal
import subprocess
import sys

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
        find_env_for_checkpoint,
        get_env_python,
    )
    from ..operations import parse_setup_kwargs
    from ..spawn import WORKER_WRAPPER, spawn_in_env
    from .common import resolve_cache_root

    root = get_root_or_exit(args)
    cache_root = resolve_cache_root(root)
    socket_path = args.socket
    checkpoint = args.checkpoint
    device = args.device

    try:
        setup_kwargs = parse_setup_kwargs(getattr(args, "kwarg", None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        env_name, _ = find_env_for_checkpoint(root, checkpoint)
    except CheckpointNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Validate environment exists before printing the startup banner
    try:
        get_env_python(root, env_name)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Starting rootstock worker:")
    print(f"  Env: {env_name}")
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
            "device": device,
            "socket_path": socket_path,
            "setup_kwargs": setup_kwargs,
        },
        cache_root=cache_root,
    ) as spec:
        proc = subprocess.Popen(spec.cmd, env=spec.env, cwd=spec.cwd)

        # Forward signals to worker
        def forward_signal(signum, frame):
            proc.send_signal(signum)

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)

        # Block until worker exits
        return proc.wait()
