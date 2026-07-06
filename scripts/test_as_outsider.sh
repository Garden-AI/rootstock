#!/usr/bin/env bash
# test_as_outsider.sh — functional test that a rootstock install is usable by
# a NON-MAINTAINER. For each built env it drives the real user path
# (RootstockCalculator -> worker subprocess -> model load -> forward pass)
# and reports PASS/FAIL per checkpoint, watching for permission errors on
# runtime writes (Triton/Inductor kernels, torch extensions, HF locks, ...).
#
# For a conclusive result this must run as a uid that does NOT own the tree,
# with no membership in the tree's group. On a login node the recommended
# harness (run by a colleague, not the maintainer) is:
#
#     unshare --user --map-user=$(id -u) \
#         ./test_as_outsider.sh /global/cfs/cdirs/m4845/rootstock \
#         --cache-root /pscratch/sd/w/wengler/rootstock-cache
#
# unshare strips supplementary groups (defeats group bits); the different uid
# defeats owner bits; what remains is the other-bits contract this verifies.
# The script warns — but does not stop — when run in a masked configuration,
# so the maintainer can still use it as a smoke test.
#
# Usage:
#   test_as_outsider.sh ROOT [--cache-root PATH] [--device DEV]
#                            [--env NAME ...] [--checkpoint ID ...] [--strace]
#
#   ROOT              rootstock install root (or use a registered cluster path)
#   --cache-root      separate model-weight cache root (Perlmutter: PSCRATCH)
#   --device          device for inference (default: cpu — works on login nodes)
#   --env NAME        only test this env (repeatable; default: all built envs)
#   --checkpoint ID   only test this checkpoint id (repeatable)
#   --strace          wrap each probe in strace and report EACCES file ops
#                     (needs strace on PATH; Linux only)
#
# Exit codes: 0 = all tested checkpoints pass, 1 = failures, 2 = usage error.

set -u

ROOT=""
CACHE_ROOT=""
DEVICE="cpu"
STRACE=0
ENV_FILTER=()
CKPT_FILTER=()

while [ $# -gt 0 ]; do
    case "$1" in
        --cache-root) CACHE_ROOT="${2:?--cache-root needs a value}"; shift ;;
        --device) DEVICE="${2:?--device needs a value}"; shift ;;
        --env) ENV_FILTER+=("${2:?--env needs a value}"); shift ;;
        --checkpoint) CKPT_FILTER+=("${2:?--checkpoint needs a value}"); shift ;;
        --strace) STRACE=1 ;;
        -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) ROOT="$1" ;;
    esac
    shift
done

if [ -z "$ROOT" ]; then
    echo "Error: install root required (see --help)." >&2
    exit 2
fi
if [ ! -d "$ROOT" ]; then
    echo "Error: cannot access $ROOT — either it does not exist or an ancestor" >&2
    echo "denies traversal. Run scripts/check_world_readable.sh from an account" >&2
    echo "that can see it." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Preflight: is this run actually conclusive?
# ---------------------------------------------------------------------------

MASKED=0

if stat -c '%u %g' / >/dev/null 2>&1; then
    read -r TREE_UID TREE_GID <<< "$(stat -c '%u %g' "$ROOT")"
else
    read -r TREE_UID TREE_GID <<< "$(stat -f '%u %g' "$ROOT")"
fi

if [ "$TREE_UID" = "$(id -u)" ]; then
    echo "WARNING: you OWN $ROOT — owner bits satisfy every access check, so"
    echo "         this run cannot prove world-readability. Have a different"
    echo "         user run it (inside 'unshare --user --map-user=\$(id -u)')."
    MASKED=1
fi
if id -G | tr ' ' '\n' | grep -qx "$TREE_GID"; then
    echo "WARNING: you are in the tree's group (gid $TREE_GID) — group bits may"
    echo "         mask failures. Run inside: unshare --user --map-user=\$(id -u)"
    MASKED=1
fi
if ! touch "${HOME}/.rootstock_outsider_write_test" 2>/dev/null; then
    echo "Error: \$HOME ($HOME) is not writable — per-user runtime caches need it." >&2
    exit 2
fi
rm -f "${HOME}/.rootstock_outsider_write_test"

if [ "$STRACE" -eq 1 ] && ! command -v strace >/dev/null 2>&1; then
    echo "WARNING: --strace requested but strace not found; continuing without it."
    STRACE=0
fi

# Cheap read canary before spending minutes loading models.
CANARY=$(find "$ROOT/envs" -name 'env_source.py' -print 2>/dev/null | head -1)
if [ -n "$CANARY" ] && ! cat "$CANARY" >/dev/null 2>&1; then
    echo "FAIL (fast): cannot read $CANARY"
    echo "The tree is not world-readable. Diagnose with:"
    echo "  scripts/check_world_readable.sh $ROOT"
    exit 1
fi

# ---------------------------------------------------------------------------
# The probe, run once per env with that env's own interpreter. It exercises
# the real code path: rootstock (installed inside the venv) resolves the env,
# spawns the worker, applies the cache-redirection env vars, loads the model,
# and runs one forward pass on the standard smoke-test structure.
# ---------------------------------------------------------------------------

WORK=$(mktemp -d "${TMPDIR:-/tmp}/rootstock_outsider.XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT

PROBE="$WORK/probe.py"
cat > "$PROBE" << 'PY'
import argparse
import sys
import time
import traceback

PERM_MARKERS = ("Errno 13", "EACCES", "Permission denied", "Read-only file system")


def perm_hint(text: str) -> str:
    hits = [line.strip() for line in text.splitlines() if any(m in line for m in PERM_MARKERS)]
    if not hits:
        return ""
    return "  PERMISSION ERROR detected:\n    " + "\n    ".join(hits[:5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--env", required=True)
    ap.add_argument("checkpoints", nargs="*")
    args = ap.parse_args()

    from pathlib import Path

    from rootstock.environment import get_model_cache_env, parse_checkpoints_dict

    root = Path(args.root)
    cache_root = Path(args.cache_root) if args.cache_root else None

    print("cache/env redirection for workers (from get_model_cache_env):")
    for key, value in sorted(get_model_cache_env(root, cache_root).items()):
        print(f"  {key}={value}")

    declared = parse_checkpoints_dict(root / "envs" / args.env / "env_source.py")

    if args.checkpoints:
        checkpoints = [c for c in args.checkpoints if c in declared]
        skipped = [c for c in args.checkpoints if c not in declared]
        for c in skipped:
            print(f"SKIP {args.env}/{c}: not declared by this env")
    else:
        # Only fetched checkpoints are testable by a non-maintainer (an
        # outsider cannot write weights into the shared cache — by design).
        checkpoints = list(declared)
        try:
            from rootstock.manifest import load_manifest

            manifest = load_manifest(root)
            if manifest is not None and args.env in manifest.environments:
                states = manifest.environments[args.env].checkpoints
                fetched = [c for c in checkpoints if c in states and states[c].fetched_at]
                for c in checkpoints:
                    if c not in fetched:
                        print(f"SKIP {args.env}/{c}: not fetched (rootstock add {c})")
                checkpoints = fetched
            else:
                print("note: no manifest entry; testing all declared checkpoints")
        except Exception as exc:  # manifest unreadable is itself worth knowing
            print(f"note: could not read manifest ({exc}); testing all declared checkpoints")

    if not checkpoints:
        print(f"SKIP {args.env}: no fetched checkpoints to test")
        return 0

    from rootstock.calculator import RootstockCalculator
    from rootstock.verify import _smoke_test_atoms

    failures = 0
    for ckpt in checkpoints:
        atoms = _smoke_test_atoms()
        start = time.monotonic()
        try:
            with RootstockCalculator(
                checkpoint=ckpt,
                root=root,
                cache_root=cache_root,
                device=args.device,
            ) as calc:
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                atoms.get_forces()
            elapsed = time.monotonic() - start
            print(f"PASS {args.env}/{ckpt}  energy={energy:.6f} eV  ({elapsed:.1f}s)")
        except Exception:
            elapsed = time.monotonic() - start
            failures += 1
            text = traceback.format_exc()
            print(f"FAIL {args.env}/{ckpt}  ({elapsed:.1f}s)")
            tail = text.strip().splitlines()
            print("  " + "\n  ".join(tail[-6:]))
            hint = perm_hint(text)
            if hint:
                print(hint)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
PY

# ---------------------------------------------------------------------------
# Drive the probe per env
# ---------------------------------------------------------------------------

want_env() {
    [ ${#ENV_FILTER[@]} -eq 0 ] && return 0
    local e
    for e in "${ENV_FILTER[@]}"; do [ "$e" = "$1" ] && return 0; done
    return 1
}

TOTAL_FAIL=0
TESTED=0

for env_dir in "$ROOT"/envs/*/; do
    [ -x "$env_dir/bin/python" ] || continue
    env_name=$(basename "$env_dir")
    want_env "$env_name" || continue
    TESTED=$((TESTED + 1))

    echo
    echo "=== env: $env_name ==="

    cmd=("$env_dir/bin/python" "$PROBE" --root "$ROOT" --env "$env_name" --device "$DEVICE")
    [ -n "$CACHE_ROOT" ] && cmd+=(--cache-root "$CACHE_ROOT")
    [ ${#CKPT_FILTER[@]} -gt 0 ] && cmd+=("${CKPT_FILTER[@]}")

    if [ "$STRACE" -eq 1 ]; then
        tracefile="$WORK/strace_$env_name"
        strace -f -qq -e trace=%file -o "$tracefile" "${cmd[@]}"
        rc=$?
        if [ -f "$tracefile" ] && grep -q 'EACCES' "$tracefile"; then
            echo "  strace: EACCES file ops (unique paths):"
            grep 'EACCES' "$tracefile" \
                | grep -oE '"[^"]+"' | sort -u | head -20 | sed 's/^/    /'
        fi
    else
        "${cmd[@]}"
        rc=$?
    fi

    [ $rc -ne 0 ] && TOTAL_FAIL=$((TOTAL_FAIL + 1))
done

echo
if [ "$TESTED" -eq 0 ]; then
    echo "Error: no built envs found under $ROOT/envs (or --env filter matched none)." >&2
    exit 2
fi

if [ "$MASKED" -eq 1 ]; then
    echo "NOTE: run was masked (owner/group access applied) — a pass here does"
    echo "      NOT prove world-readability. See warnings above."
fi

if [ "$TOTAL_FAIL" -gt 0 ]; then
    echo "RESULT: FAIL — $TOTAL_FAIL of $TESTED env(s) had failures."
    exit 1
fi
echo "RESULT: OK — all tested envs passed as an outside user."
exit 0
