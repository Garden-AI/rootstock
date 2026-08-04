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

    rootstock sync [<source-dir>] [--root <path> | --cluster <name>] [--dry-run]
        Converge the install to its declared state: build missing/changed
        envs, download and verify missing/stale checkpoints, in parallel
        phases. Idempotent — re-run to retry whatever failed.
            rootstock sync --cluster delta --dry-run
            rootstock sync ./environments/ --jobs 8
            rootstock sync --rebuild                 # after a CLI version bump
            rootstock sync --phases build,download   # login node (no GPU)

    rootstock prune [<source-dir>] [--root <path> | --cluster <name>] [--dry-run] [--yes]
        The subtractive half of sync: remove built envs with no registered
        source, checkpoint records (and their unshared weight files) no
        source declares, and internal garbage (.build/.trash leftovers,
        orphaned interpreters, stale lockfiles, the uv cache). Plan-confirm
        by default; batch jobs pass --yes.
            rootstock prune --cluster delta --dry-run
            rootstock prune ./environments/ --yes    # retire everything not declared here
            rootstock prune --gc-only --yes          # internal garbage only

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
            rootstock setup-perms --cluster perlmutter --group m5268 --apply

    rootstock check-perms [<root>] [--cluster <name>] [--group <group>] [--json]
        Read-only check that the install root, split cache root, and their
        ancestor directories satisfy the shared-install permission recipe.
        Exits 0 when clean, 1 when issues are found:
            rootstock check-perms --cluster perlmutter --group m5268

    rootstock usage report [--root <path>] [--json]
    rootstock usage compact [--root <path>]
    rootstock usage push [--root <path>] [--dry-run]
        Aggregate (read-only), compact, or push to the dashboard backend the
        anonymous usage-record spool at {cache_root}/usage/. The spool is
        provisioned by setup-perms; without it, usage collection is off.
"""

import argparse
import os
import sys

from . import __version__
from .batch import DEFAULT_MIN_AGE_HOURS
from .commands import (
    cmd_add,
    cmd_benchmark,
    cmd_check_perms,
    cmd_init,
    cmd_install,
    cmd_list,
    cmd_manifest_init,
    cmd_manifest_push,
    cmd_manifest_show,
    cmd_new_env,
    cmd_prune,
    cmd_resolve,
    cmd_serve,
    cmd_setup_perms,
    cmd_smoke_test,
    cmd_status,
    cmd_sync,
    cmd_usage_compact,
    cmd_usage_push,
    cmd_usage_report,
)
from .commands.common import ROOTSTOCK_ROOT_ENV
from .config import DEFAULT_CONFIG_FILE
from .manifest import ManifestError
from .verify import DEFAULT_VERIFY_TIMEOUT

_VERIFY_TIMEOUT_HELP = (
    "Seconds allowed per verification — worker startup (env import + model "
    f"load) plus one forward pass (default: {DEFAULT_VERIFY_TIMEOUT:.0f}). "
    "Raise on slow filesystems where cold reads of large checkpoints "
    "exceed the default"
)


def main():
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
        "--cache-root",
        help=(
            "Filesystem for model weights when it differs from the install root "
            "(recorded in {root}/layout.json). Prompted for if omitted. This is "
            "a deployment-time choice: changing it later means editing "
            "layout.json and moving the weights."
        ),
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
        "--verify-timeout",
        type=float,
        default=DEFAULT_VERIFY_TIMEOUT,
        metavar="SECONDS",
        help=_VERIFY_TIMEOUT_HELP,
    )
    add_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the verify phase (download only). Login-node escape hatch.",
    )
    add_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run the download even if the manifest records the checkpoint "
            "as fetched — repairs a cache file that has gone missing. Cheap "
            "when the cache is intact (cache hit, no transfer)."
        ),
    )
    add_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    add_parser.add_argument(
        "--cluster",
        help=(
            "Cluster this machine is (e.g. 'sophia'). Required to verify on "
            "a shared install — sophia/polaris serve one root, and results "
            "must be recorded for the right machine. Also picks the right "
            "env among cluster-specific variants (CLUSTERS). Single-cluster "
            "installs need no flag."
        ),
    )
    add_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend",
    )
    add_parser.set_defaults(func=cmd_add)

    # sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Converge an install to its declared state (batch build/download/verify)",
        description=(
            "Plan and execute the delta between declared state (registered env "
            "sources, optionally overlaid by a staging directory, plus their "
            "CHECKPOINTS tables) and actual state (built envs + manifest). "
            "Idempotent: re-running after a failure retries only what didn't "
            "converge. Runs work in parallel per phase; verification "
            "concurrency is bounded separately (each verify loads a model "
            "onto the GPU)."
        ),
    )
    sync_parser.add_argument(
        "source_dir",
        nargs="?",
        help=(
            "Optional directory of env source files (*.py) to register/update "
            "before converging; defaults to the root's registered environments"
        ),
    )
    sync_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    sync_parser.add_argument(
        "--cluster",
        help=(
            "Resolve the root from the cluster registry instead of --root; "
            "also names the machine verification results are recorded for "
            "(required to verify on a shared install like sophia/polaris)"
        ),
    )
    sync_parser.add_argument(
        "--env",
        action="append",
        metavar="NAME",
        help="Limit to this environment (and its checkpoints); repeatable",
    )
    sync_parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="ID",
        help="Limit to this checkpoint id; repeatable",
    )
    sync_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force-rebuild the selected environments (e.g. after a CLI version bump)",
    )
    sync_parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Re-resolve env lockfiles during rebuilds instead of honoring them",
    )
    sync_parser.add_argument(
        "--phases",
        default=",".join(("build", "download", "verify")),
        help=(
            "Comma-separated subset of build,download,verify (default: all). "
            "E.g. build,download on a login node, then verify in a GPU job"
        ),
    )
    sync_parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Build/download parallelism (default: 4)",
    )
    sync_parser.add_argument(
        "--verify-jobs",
        type=int,
        default=1,
        help=(
            "Verify parallelism (default: 1 — each verify loads a model onto "
            "the GPU). With --device cuda, N workers round-robin cuda:0..N-1"
        ),
    )
    sync_parser.add_argument(
        "--verify-timeout",
        type=float,
        default=DEFAULT_VERIFY_TIMEOUT,
        metavar="SECONDS",
        help=_VERIFY_TIMEOUT_HELP,
    )
    sync_parser.add_argument("--device", default="cuda", help="Device for verify (default: cuda)")
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without doing anything",
    )
    sync_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the plan and results as JSON on stdout (progress goes to stderr)",
    )
    sync_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling new work after the first failure (default: keep going)",
    )
    sync_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend",
    )
    sync_parser.add_argument(
        "--no-perm-check",
        action="store_true",
        help="Skip the up-front shared-install permission check",
    )
    sync_parser.set_defaults(func=cmd_sync)

    # prune command
    prune_parser = subparsers.add_parser(
        "prune",
        help="Remove undeclared envs/checkpoints and internal garbage (sync's subtractive half)",
        description=(
            "Plan and execute the removal of actual state not covered by "
            "declared state: built envs with no registered source, manifest "
            "checkpoint records no source declares (plus their weight files, "
            "refcounted against surviving checkpoints), and internal garbage "
            "(.build/.trash leftovers, orphaned interpreters, stale "
            "lockfiles, the uv cache). Prints the plan and asks for "
            "confirmation before deleting anything; idempotent and safe to "
            "re-run after a failure. Don't run while a sync is in flight."
        ),
    )
    prune_parser.add_argument(
        "source_dir",
        nargs="?",
        help=(
            "Optional directory declaring the *complete* desired set of env "
            "sources (*.py): anything registered, built, or fetched beyond it "
            "is pruned — including registered source files. An empty directory "
            "declares zero environments. Defaults to the root's registered "
            "environments (if that dir doesn't exist, nothing is declared and "
            "only internal garbage is collected)."
        ),
    )
    prune_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    prune_parser.add_argument(
        "--cluster",
        help="Resolve the root from the cluster registry instead of --root",
    )
    prune_parser.add_argument(
        "--env",
        action="append",
        metavar="NAME",
        help="Limit to this environment (and its checkpoints); repeatable",
    )
    prune_parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="ID",
        help=(
            "Limit pruning to this checkpoint id (env pruning is skipped "
            "unless --env is also given); repeatable"
        ),
    )
    prune_parser.add_argument(
        "--gc-only",
        action="store_true",
        help="Collect internal garbage only; never touches envs or checkpoints",
    )
    prune_parser.add_argument(
        "--deep",
        action="store_true",
        help=(
            "Also delete unattributed cache/home contents (anything no "
            "checkpoint's weight record claims). Only sensible once weight "
            "records have been backfilled — run a smoke-test first"
        ),
    )
    prune_parser.add_argument(
        "--min-age",
        type=float,
        default=DEFAULT_MIN_AGE_HOURS,
        metavar="HOURS",
        help=(
            "Leave .build/.trash/interpreter entries (and --deep candidates) "
            "younger than this alone — they may belong to work running on "
            f"another node (default: {DEFAULT_MIN_AGE_HOURS:g})"
        ),
    )
    prune_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (for batch jobs)",
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without deleting anything",
    )
    prune_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the plan and results as JSON on stdout (progress goes to stderr)",
    )
    prune_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling new deletions after the first failure (default: keep going)",
    )
    prune_parser.add_argument(
        "--no-push",
        action="store_true",
        help="Don't push manifest to backend",
    )
    prune_parser.add_argument(
        "--no-perm-check",
        action="store_true",
        help="Skip the up-front shared-install permission check",
    )
    prune_parser.set_defaults(func=cmd_prune)

    # smoke-test command
    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Re-verify checkpoints already registered in the manifest",
        description=(
            "Re-verify checkpoints by running a forward pass on each. Never downloads. "
            "Canonical checkpoints always use setup_kwargs={} — checkpoints requiring "
            "non-default kwargs may show as failing here even if they work in practice. "
            "Each run also exercises every '<family>:custom' entry by re-loading a "
            "same-family checkpoint's cached weights via the weights= path and "
            "requiring identical results."
        ),
    )
    smoke_parser.add_argument("--env", help="Filter to a single environment")
    smoke_parser.add_argument(
        "--checkpoint",
        help=(
            "Filter to a single checkpoint (requires --env); a '<family>:custom' id "
            "runs just that custom-weights leg and its baseline"
        ),
    )
    smoke_parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    smoke_parser.add_argument(
        "--verify-timeout",
        type=float,
        default=DEFAULT_VERIFY_TIMEOUT,
        metavar="SECONDS",
        help=_VERIFY_TIMEOUT_HELP,
    )
    smoke_parser.add_argument("--json", action="store_true", help="Emit a JSON summary")
    smoke_parser.add_argument(
        "--cluster",
        help=(
            "Cluster this machine is (e.g. 'sophia'). Required on a shared "
            "install — sophia/polaris serve one root, and results must be "
            "recorded (and pushed) for the right machine. Single-cluster "
            "installs need no flag."
        ),
    )
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
    # --root is how every other command spells it; accept both rather than
    # failing with "unrecognized arguments" on the obvious guess.
    setup_perms_parser.add_argument(
        "--root",
        dest="root_flag",
        help="Install root path (same as the positional argument)",
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
        help="Project group that owns the install (e.g., m5268)",
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
    setup_perms_parser.add_argument(
        "--no-usage-spool",
        action="store_true",
        help=(
            "Skip provisioning the world-writable usage-record spool "
            "({cache_root}/usage, mode 1777); its absence keeps usage "
            "collection off for this install"
        ),
    )
    setup_perms_parser.add_argument(
        "--usage-dir",
        help=(
            "Redirect the usage-record spool: create the real 1777 directory "
            "at this path and symlink {cache_root}/usage to it. For clusters "
            "where maintainer write access to the install is temporary — put "
            "the spool somewhere you keep control of (e.g. under your home)"
        ),
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
        "--root",
        dest="root_flag",
        help="Install root path (same as the positional argument)",
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
        help=(
            "Canonical checkpoint id (e.g., 'mace-mp-0-medium', 'uma-s-1p1'), "
            "or a ':custom' entry (e.g. 'uma:custom') with --weights to run "
            "your own fine-tune"
        ),
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
    serve_parser.add_argument(
        "--weights",
        metavar="PATH",
        help=(
            "Path to your own weights file (must be visible from the compute "
            "nodes). Required with — and only valid with — a ':custom' "
            "checkpoint id; loaded via the env's setup_from_path hook."
        ),
    )
    serve_parser.add_argument(
        "--cluster",
        help=(
            "Cluster this machine is (e.g. 'polaris'). Only needed on shared "
            "installs whose envs declare cluster-specific variants (CLUSTERS) "
            "— picks the variant serving this machine."
        ),
    )
    serve_parser.set_defaults(func=cmd_serve)

    # benchmark command. Everything after the subcommand is forwarded to the
    # benchmark's own parser via the parse_known_args leftovers at the bottom
    # of main() — argparse.REMAINDER cannot capture leading options (--list as
    # the first forwarded arg errors), so the subparser declares no arguments
    # and add_help=False lets --help through to the benchmark parser.
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
    manifest_show_parser.set_defaults(func=cmd_manifest_show)

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
    manifest_push_parser.set_defaults(func=cmd_manifest_push)

    # manifest init
    manifest_init_parser = manifest_subparsers.add_parser(
        "init",
        help="Initialize manifest for the cluster(s) an install serves",
    )
    manifest_init_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    manifest_init_parser.add_argument(
        "--cluster",
        required=True,
        action="append",
        help=(
            "Cluster name (e.g., delta). Repeatable for shared installs "
            "(--cluster sophia --cluster polaris); registry siblings of the "
            "root are included automatically."
        ),
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
    manifest_init_parser.set_defaults(func=cmd_manifest_init)

    # usage command
    usage_parser = subparsers.add_parser(
        "usage",
        help="Report on or compact the usage-record spool",
        description=(
            "Maintainer-side view of the anonymous usage records that "
            "calculator sessions spool to {cache_root}/usage/. 'report' "
            "aggregates read-only; 'compact' folds raw records into "
            "per-month rollup files; 'push' sends the aggregated rollup "
            "rows to the dashboard backend."
        ),
    )
    usage_subparsers = usage_parser.add_subparsers(
        dest="usage_action",
        required=True,
    )

    usage_report_parser = usage_subparsers.add_parser(
        "report",
        help="Aggregate the spool (read-only)",
    )
    usage_report_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    usage_report_parser.add_argument(
        "--cache-root",
        help="Cache root override (default: the install's own declaration)",
    )
    usage_report_parser.add_argument("--json", action="store_true", help="Output as JSON")
    usage_report_parser.set_defaults(func=cmd_usage_report)

    usage_compact_parser = usage_subparsers.add_parser(
        "compact",
        help="Fold raw records into per-month rollup files",
    )
    usage_compact_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    usage_compact_parser.add_argument(
        "--cache-root",
        help="Cache root override (default: the install's own declaration)",
    )
    usage_compact_parser.set_defaults(func=cmd_usage_compact)

    usage_push_parser = usage_subparsers.add_parser(
        "push",
        help="Push aggregated rollup rows to the dashboard backend",
    )
    usage_push_parser.add_argument(
        "--root",
        default=os.environ.get(ROOTSTOCK_ROOT_ENV),
        help=f"Root directory (default: ${ROOTSTOCK_ROOT_ENV})",
    )
    usage_push_parser.add_argument(
        "--cache-root",
        help="Cache root override (default: the install's own declaration)",
    )
    usage_push_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the endpoint URL and payload instead of pushing",
    )
    usage_push_parser.set_defaults(func=cmd_usage_push)

    # parse_known_args instead of parse_args so `rootstock benchmark ...` can
    # forward arbitrary flags to the benchmark's own parser. Every other
    # command keeps strict parsing via the explicit error below.
    args, extra = parser.parse_known_args()
    if args.command == "benchmark":
        sys.exit(cmd_benchmark(extra))
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        sys.exit(args.func(args))
    except ManifestError as exc:
        # A corrupt or incompatible manifest is a data-integrity stop, not a
        # crash: print the diagnosis cleanly instead of a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
