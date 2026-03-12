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
    from ..environment import EnvironmentManager

    root = get_root_or_exit(args)
    env_name = f"{args.model}_env"
    socket_path = args.socket
    checkpoint = args.checkpoint
    device = args.device

    # Create environment manager and validate environment exists
    env_mgr = EnvironmentManager(root=root)
    try:
        env_mgr.get_env_python(env_name)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Generate wrapper script
    wrapper_path = env_mgr.generate_wrapper(
        env_name=env_name,
        model=checkpoint,
        device=device,
        socket_path=socket_path,
    )

    # Get spawn command and environment
    cmd = env_mgr.get_spawn_command(env_name, wrapper_path)
    env = env_mgr.get_environment_variables()

    print("Starting rootstock worker:")
    print(f"  Model: {args.model} (env: {env_name})")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device: {device}")
    print(f"  Socket: {socket_path}")

    # Spawn worker subprocess
    proc = subprocess.Popen(cmd, env=env)

    # Forward signals to worker
    def forward_signal(signum, frame):
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    # Block until worker exits
    try:
        rc = proc.wait()
    finally:
        env_mgr.cleanup()

    return rc
