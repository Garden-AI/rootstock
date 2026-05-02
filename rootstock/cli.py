"""
Rootstock CLI.

The --root flag specifies the rootstock root directory. If not provided,
the ROOTSTOCK_ROOT environment variable is used.

Commands:
    rootstock init
        Interactive setup of configuration and directory structure.

    rootstock new-env <name> [-o <path>] [--force]
        Create a new environment file from template:
            rootstock new-env mace
            rootstock new-env mace -o ./environments/mace_env.py

    rootstock install <source> [--root <path>] [--models m1,m2] [--force]
        Install from file (validates, registers, builds):
            rootstock install ./mace_env.py --root /vol/rootstock
        Install all environments from a directory:
            rootstock install ./environments/ --root /vol/rootstock
        Rebuild existing environment by name:
            rootstock install mace_env --root /vol/rootstock --force

    rootstock status [--root <path>]
    rootstock list [--root <path>]
    rootstock serve <model> [--root <path>] --socket <path> --checkpoint <name> [--device <dev>]
    rootstock resolve --cluster <name> [--json]
"""

import argparse
import os
import sys

from .commands import (
    cmd_add,
    cmd_init,
    cmd_install,
    cmd_list,
    cmd_manifest,
    cmd_new_env,
    cmd_resolve,
    cmd_serve,
    cmd_smoke_test,
    cmd_status,
)
from .commands.common import ROOTSTOCK_ROOT_ENV
from .config import DEFAULT_CONFIG_FILE


def main():
    parser = argparse.ArgumentParser(
        prog="rootstock",
        description="Rootstock MLIP environment manager",
        epilog=f"Config file: {DEFAULT_CONFIG_FILE}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Interactive setup of rootstock configuration",
        description="Guided setup for root directory, maintainer info, and API credentials.",
    )
    init_parser.add_argument(
        "--skip-dirs",
        action="store_true",
        help="Skip creating directory structure",
    )
    init_parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip initializing manifest",
    )
    init_parser.set_defaults(func=cmd_init)

    # new-env command
    new_env_parser = subparsers.add_parser(
        "new-env",
        help="Create a new environment file from template",
        description="Scaffold a new environment source file with the required structure.",
    )
    new_env_parser.add_argument(
        "name",
        help="Environment name (e.g., 'mace' or 'mace_env')",
    )
    new_env_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: ./<name>_env.py)",
    )
    new_env_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )
    new_env_parser.set_defaults(func=cmd_new_env)

    # install command
    install_parser = subparsers.add_parser(
        "install",
        help="Install environment(s) from file, directory, or rebuild by name",
        description=(
            "Install environment(s) from a file, directory, or rebuild by name. "
            "File: validates, registers, and builds a single environment. "
            "Directory: installs all *.py environment files. "
            "Name: rebuilds an existing registered environment."
        ),
    )
    install_parser.add_argument(
        "source",
        help="File path, directory, or env name (e.g., ./mace_env.py, ./environments/, mace_env)",
    )
    install_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    # --models is intentionally still parsed so we can emit a clear migration
    # message in cmd_install. Removed in v0.8.0; use `rootstock add` instead.
    install_parser.add_argument(
        "--models",
        help=argparse.SUPPRESS,
    )
    install_parser.add_argument(
        "--force", action="store_true", help="Update registration and/or rebuild if exists"
    )
    install_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    install_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend (useful during development)",
    )
    install_parser.set_defaults(func=cmd_install)

    # add command
    add_parser = subparsers.add_parser(
        "add",
        help="Download and verify a checkpoint for an installed environment",
        description=(
            "Idempotent download-or-verify. Skips download if already fetched. "
            "Use --no-verify on login nodes without GPUs."
        ),
    )
    add_parser.add_argument("env", help="Environment name (e.g., 'mace', 'mace_env')")
    add_parser.add_argument(
        "checkpoint",
        help="Checkpoint identifier (e.g., 'medium', 'uma-s-1p1')",
    )
    add_parser.add_argument(
        "--kwarg",
        action="append",
        metavar="KEY=VAL",
        help=(
            "Extra kwarg passed to setup() (repeatable). Value is JSON-decoded "
            "first, then falls back to a string. E.g., --kwarg task=omat --kwarg charge=-1"
        ),
    )
    add_parser.add_argument("--device", default="cuda", help="Device for verify (default: cuda)")
    add_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the verify phase (download only). Login-node escape hatch.",
    )
    add_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    add_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend",
    )
    add_parser.set_defaults(func=cmd_add)

    # smoke-test command
    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Re-verify checkpoints already registered in the manifest",
        description=(
            "Re-verify checkpoints by running a forward pass on each. Never downloads. "
            "Always uses setup_kwargs={} — checkpoints requiring non-default kwargs "
            "may show as failing here even if they work in practice."
        ),
    )
    smoke_parser.add_argument("--env", help="Filter to a single environment")
    smoke_parser.add_argument(
        "--checkpoint", help="Filter to a single checkpoint (requires --env)"
    )
    smoke_parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    smoke_parser.add_argument("--json", action="store_true", help="Emit a JSON summary")
    smoke_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    smoke_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend",
    )
    smoke_parser.set_defaults(func=cmd_smoke_test)

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of rootstock installation",
        description="Show environment sources, built environments, and cache sizes.",
    )
    status_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output the manifest as JSON (with computed verified_current per checkpoint)",
    )
    status_parser.set_defaults(func=cmd_status)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List registered environments",
        description="List all environment files in the shared environments directory.",
    )
    list_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    list_parser.set_defaults(func=cmd_list)

    # resolve command
    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve cluster configuration",
        description="Look up the root directory for a known cluster.",
    )
    resolve_parser.add_argument("--cluster", required=True, help="Cluster name")
    resolve_parser.add_argument("--json", action="store_true", help="Output as JSON")
    resolve_parser.set_defaults(func=cmd_resolve)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a worker for an external i-PI server",
        description="Start a rootstock worker that connects to a Unix socket.",
    )
    serve_parser.add_argument("model", help="Model family (e.g., mace, uma, tensornet)")
    serve_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    serve_parser.add_argument("--socket", required=True, help="Unix socket path to connect to")
    serve_parser.add_argument("--checkpoint", required=True, help="Checkpoint/weights name")
    serve_parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    serve_parser.add_argument(
        "--kwarg",
        action="append",
        metavar="KEY=VAL",
        help=(
            "Extra kwarg passed to setup() (repeatable). Same JSON-then-string "
            "parsing as 'rootstock add'."
        ),
    )
    serve_parser.set_defaults(func=cmd_serve)

    # manifest command
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Manage installation manifest",
        description="Manage the manifest that tracks installation state.",
    )
    manifest_subparsers = manifest_parser.add_subparsers(
        dest="manifest_action",
        required=True,
    )

    # manifest show
    manifest_show_parser = manifest_subparsers.add_parser(
        "show",
        help="Show current manifest",
    )
    manifest_show_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    manifest_show_parser.add_argument("--json", action="store_true", help="Output as JSON")
    manifest_show_parser.set_defaults(func=cmd_manifest)

    # manifest push
    manifest_push_parser = manifest_subparsers.add_parser(
        "push",
        help="Push manifest to backend",
    )
    manifest_push_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    manifest_push_parser.set_defaults(func=cmd_manifest)

    # manifest init
    manifest_init_parser = manifest_subparsers.add_parser(
        "init",
        help="Initialize manifest for a cluster",
    )
    manifest_init_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    manifest_init_parser.add_argument(
        "--cluster",
        required=True,
        help="Cluster name (e.g., della, modal)",
    )
    manifest_init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest",
    )
    manifest_init_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend (useful during development)",
    )
    manifest_init_parser.set_defaults(func=cmd_manifest)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
