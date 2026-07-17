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
            rootstock new-env mace -o ./environments/mace.py

    rootstock install <source> [--root <path>] [--force]
        Install from file (validates, registers, builds):
            rootstock install ./mace.py --root /vol/rootstock
        Install all environments from a directory:
            rootstock install ./environments/ --root /vol/rootstock
        Rebuild existing environment by name:
            rootstock install mace --root /vol/rootstock --force

    rootstock add <checkpoint-id> [--root <path>] [--device <dev>] [--kwarg KEY=VAL ...]
        Resolve the env that hosts <checkpoint-id> from the installed envs,
        then download and verify the weights.
    rootstock add --list [--root <path>]
        List every canonical checkpoint id that add accepts, grouped by env.

    rootstock benchmark [--root <path>] [--checkpoints <id> ...] [--devices cuda cpu] [--list]
        Measure i-PI IPC overhead: RootstockCalculator vs. the same calculator
        called directly inside its pre-built env. `--list` shows installed ids.

    rootstock status [--root <path>]
    rootstock list [--root <path>]
    rootstock serve <checkpoint-id> [--root <path>] --socket <path> [--device <dev>]
    rootstock resolve --cluster <name> [--json]
    rootstock setup-perms [<root>] [--cluster <name>] --group <group> [--apply] [--retrofit]
        Render (dry-run) or apply the world-readable shared-install permission
        recipe for an install root (and split cache root):
            rootstock setup-perms --cluster perlmutter --group m4845 --apply

    rootstock check-perms [<root>] [--cluster <name>] [--group <group>] [--json]
        Read-only check that the install root, split cache root, and their
        ancestor directories satisfy the shared-install permission recipe.
        Exits 0 when clean, 1 when issues are found:
            rootstock check-perms --cluster perlmutter --group m4845
"""

import argparse
import os
import sys

from . import __version__
from .commands import (
    cmd_add,
    cmd_benchmark,
    cmd_check_perms,
    cmd_init,
    cmd_install,
    cmd_list,
    cmd_manifest,
    cmd_new_env,
    cmd_resolve,
    cmd_serve,
    cmd_setup_perms,
    cmd_smoke_test,
    cmd_status,
)
from .commands.common import ROOTSTOCK_ROOT_ENV
from .config import DEFAULT_CONFIG_FILE


def main():
    # `rootstock benchmark ...` forwards everything after the subcommand to the
    # benchmark's own argument parser. Intercept it before the main parser runs,
    # since argparse can't cleanly pass arbitrary flags through a subparser.
    argv = sys.argv[1:]
    if argv and argv[0] == "benchmark":
        sys.exit(cmd_benchmark(argv[1:]))

    parser = argparse.ArgumentParser(
        prog="rootstock",
        description="Rootstock MLIP environment manager",
        epilog=f"Config file: {DEFAULT_CONFIG_FILE}",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
        help="Environment name (e.g., 'mace')",
    )
    new_env_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: ./<name>.py)",
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
        help="File path, directory, or env name (e.g., ./mace.py, ./environments/, mace)",
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
    install_parser.add_argument(
        "--upgrade",
        action="store_true",
        help=(
            "Re-resolve all dependencies to the latest allowed versions instead of "
            "honoring the environment's existing lockfile"
        ),
    )
    install_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    install_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend (useful during development)",
    )
    install_parser.add_argument(
        "--no-perm-check",
        action="store_true",
        help="Skip the up-front shared-install permission check",
    )
    install_parser.set_defaults(func=cmd_install)

    # add command
    add_parser = subparsers.add_parser(
        "add",
        help="Download and verify a checkpoint by canonical id",
        description=(
            "Idempotent download-or-verify. Resolves the hosting env from the "
            "installed envs by matching the canonical checkpoint id against each "
            "env's CHECKPOINTS dict. Skips download if already fetched. "
            "Use --no-verify on login nodes without GPUs. "
            "Pass --list to see every checkpoint id that can be added."
        ),
    )
    add_parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Canonical checkpoint id (e.g., 'mace-mp-0-medium', 'uma-s-1p1')",
    )
    add_parser.add_argument(
        "--list",
        action="store_true",
        help="List canonical checkpoint ids that add accepts (grouped by env) and exit",
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
    smoke_parser.add_argument("--checkpoint", help="Filter to a single checkpoint (requires --env)")
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
    status_parser.add_argument(
        "--sizes",
        action="store_true",
        help=(
            "Also compute per-directory cache sizes (full recursive stat of the "
            "model cache — can take minutes on Lustre/GPFS for large caches)"
        ),
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

    # setup-perms command
    setup_perms_parser = subparsers.add_parser(
        "setup-perms",
        help="Render or apply shared-install permissions",
        description=(
            "Render (dry-run, default) or apply the permission recipe for a "
            "world-readable shared install: setgid + group-write for "
            "co-maintainers, world read+traverse, and default ACLs so new files "
            "inherit. Pass --apply to execute the commands after confirmation."
        ),
    )
    setup_perms_parser.add_argument(
        "root",
        nargs="?",
        help="Install root path (omit when using --cluster)",
    )
    setup_perms_parser.add_argument(
        "--cache-root",
        help="Cache root path, when on a separate filesystem from the install root",
    )
    setup_perms_parser.add_argument(
        "--cluster",
        help="Resolve install and cache roots from the cluster registry",
    )
    setup_perms_parser.add_argument(
        "--group",
        required=True,
        help="Project group that owns the install (e.g., m4845)",
    )
    setup_perms_parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the commands (after a confirmation prompt)",
    )
    setup_perms_parser.add_argument(
        "--retrofit",
        action="store_true",
        help="Also apply recursively so existing files become world-readable",
    )
    setup_perms_parser.set_defaults(func=cmd_setup_perms)

    # check-perms command
    check_perms_parser = subparsers.add_parser(
        "check-perms",
        help="Check shared-install permissions (read-only)",
        description=(
            "Read-only check that the install root, split cache root, and their "
            "ancestor directories satisfy the shared-install permission recipe "
            "(world read+traverse, setgid, default ACLs, co-maintainer group ACL, "
            "no mask clamp). Never modifies anything. "
            "Exit codes: 0 = clean, 1 = issues found, 2 = usage error."
        ),
    )
    check_perms_parser.add_argument(
        "root",
        nargs="?",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Install root path (default: ${ROOTSTOCK_ROOT_ENV}; or use --cluster)",
    )
    check_perms_parser.add_argument(
        "--cache-root",
        help="Cache root path, when on a separate filesystem from the install root",
    )
    check_perms_parser.add_argument(
        "--cluster",
        help="Resolve install and cache roots from the cluster registry",
    )
    check_perms_parser.add_argument(
        "--group",
        help=(
            "Project group expected in the co-maintainer ACL "
            "(default: the install root's owning group)"
        ),
    )
    check_perms_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report",
    )
    check_perms_parser.set_defaults(func=cmd_check_perms)

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a worker for a server speaking the rootstock i-PI dialect",
        description=(
            "Start a rootstock worker that connects to a Unix socket. The peer "
            "must speak the rootstock i-PI dialect (JSON INIT payload, re-sent "
            "every force cycle) — a standard i-PI server cannot drive it; see "
            "the 'Deviations from standard i-PI' section of the architecture "
            "docs. Known dialect peers: RootstockCalculator and the LAMMPS "
            "fix_rootstock."
        ),
    )
    serve_parser.add_argument(
        "checkpoint",
        help="Canonical checkpoint id (e.g., 'mace-mp-0-medium', 'uma-s-1p1')",
    )
    serve_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    serve_parser.add_argument("--socket", required=True, help="Unix socket path to connect to")
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

    # benchmark command. Registered only so it appears in `rootstock --help`;
    # the actual dispatch happens in the early intercept at the top of main(),
    # which forwards all following args to the benchmark's own parser. (argparse
    # REMAINDER can't reliably capture leading options, so we bypass it.)
    subparsers.add_parser(
        "benchmark",
        help="Measure i-PI IPC overhead vs. in-env direct calls",
        description=(
            "Compare RootstockCalculator against the same calculator called "
            "directly inside its pre-built env. See `rootstock benchmark --help` "
            "for options."
        ),
        add_help=False,
    )

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
