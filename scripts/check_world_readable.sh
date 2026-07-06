#!/usr/bin/env bash
# check_world_readable.sh — static audit that a directory tree is usable by
# ANY user on the machine (the "world-readable contract"):
#
#   * every ancestor of the tree grants o+x (world can traverse down to it)
#   * every directory in the tree grants o+rx
#   * every file grants o+r; owner-executable files also grant o+x
#   * symlink targets resolve and satisfy the same rules (a world-readable
#     symlink to a 700 target still fails)
#   * no per-user ACL entries or mask clamps silently override the mode bits
#
# This complements (does not replace) a functional test as a different uid:
# static checks catch owner-bit masking that "try it yourself" never can.
# See scripts/test_as_outsider.sh for the functional half.
#
# Usage:
#   check_world_readable.sh TREE [--no-acl] [--no-prefix]
#                                [--expect-group NAME] [--max-report N]
#   check_world_readable.sh --self-test
#
#   --no-acl          skip the getfacl scan (it is the slow part on big trees)
#   --no-prefix       skip the ancestor-directory walk
#   --expect-group G  named-group ACL entries for G are expected (setup-perms
#                     adds them) and not reported
#   --max-report N    cap printed paths per category (default 50; 0 = no cap).
#                     Totals are always printed — nothing is silently dropped.
#   --self-test       build a deliberately broken fixture tree in $TMPDIR,
#                     run this script against it, and verify every check fires.
#
# Exit codes: 0 = tree satisfies the contract, 1 = violations found,
#             2 = usage/environment error.
#
# Portability: works with GNU (Linux login nodes) and BSD (macOS, for local
# fixture testing) stat/find. NERSC caveat: getfacl does not show GPFS NFSv4
# ACLs — if the filesystem uses them (CFS can), spot-check a few files with
# `mmgetacl` by hand; there is no cheap recursive equivalent.

set -u

MAX_REPORT=50
DO_ACL=1
DO_PREFIX=1
EXPECT_GROUP=""
TREE=""
SELF_TEST=0

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --no-acl) DO_ACL=0 ;;
        --no-prefix) DO_PREFIX=0 ;;
        --expect-group) EXPECT_GROUP="${2:?--expect-group needs a value}"; shift ;;
        --max-report) MAX_REPORT="${2:?--max-report needs a value}"; shift ;;
        --self-test) SELF_TEST=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *) TREE="$1" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Portable stat helpers
# ---------------------------------------------------------------------------

if stat -c '%a' / >/dev/null 2>&1; then
    mode_of() { stat -c '%a' -- "$1" 2>/dev/null; }
    lsmode_of() { stat -c '%A %U %G' -- "$1" 2>/dev/null; }
else
    mode_of() { stat -f '%Lp' -- "$1" 2>/dev/null; }
    lsmode_of() { stat -f '%Sp %Su %Sg' -- "$1" 2>/dev/null; }
fi

# Other-permission digit (last octal digit) of a path's mode.
other_digit() {
    local m
    m=$(mode_of "$1") || return 1
    printf '%s' "${m: -1}"
}

has_o_r() { local d; d=$(other_digit "$1") || return 1; [ $((d & 4)) -ne 0 ]; }
has_o_x() { local d; d=$(other_digit "$1") || return 1; [ $((d & 1)) -ne 0 ]; }

resolve_path() {
    readlink -f -- "$1" 2>/dev/null \
        || python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

# ---------------------------------------------------------------------------
# Violation accumulation and reporting
# ---------------------------------------------------------------------------

WORK=$(mktemp -d "${TMPDIR:-/tmp}/check_world_readable.XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT

add_violation() { # add_violation CATEGORY "detail line"
    printf '%s\n' "$2" >> "$WORK/$1"
}

report_category() { # report_category CATEGORY "header" "fix hint"
    local f="$WORK/$1" n
    [ -s "$f" ] || return 0
    n=$(wc -l < "$f" | tr -d ' ')
    echo
    echo "VIOLATION [$1] $2 — $n path(s):"
    if [ "$MAX_REPORT" -gt 0 ] && [ "$n" -gt "$MAX_REPORT" ]; then
        head -n "$MAX_REPORT" "$f" | sed 's/^/  /'
        echo "  ... and $((n - MAX_REPORT)) more (rerun with --max-report 0 to list all)"
    else
        sed 's/^/  /' "$f"
    fi
    echo "  fix: $3"
    FAILED=1
}

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

check_prefix() {
    # Every ancestor directory needs o+x or nobody outside the owner/group can
    # even reach the tree, no matter how open the tree itself is (namei -l
    # shows the same walk).
    local p
    p=$(dirname "$TREE")
    while :; do
        if ! has_o_x "$p"; then
            add_violation prefix "$(lsmode_of "$p") $p"
        fi
        [ "$p" = "/" ] && break
        p=$(dirname "$p")
    done
}

check_tree_modes() {
    # Octal -perm forms work in both GNU and BSD find:
    #   -perm -0004 = other-read set, -perm -0001 = other-execute set.
    find "$TREE" -type d \( ! -perm -0004 -o ! -perm -0001 \) 2>>"$WORK/errors" \
        | while IFS= read -r p; do add_violation dirs "$(lsmode_of "$p") $p"; done

    find "$TREE" -type f ! -perm -0004 2>>"$WORK/errors" \
        | while IFS= read -r p; do add_violation files "$(lsmode_of "$p") $p"; done

    # Owner-executable but not world-executable: bin/python et al. would be
    # readable yet not runnable by other users.
    find "$TREE" -type f -perm -0100 ! -perm -0001 2>>"$WORK/errors" \
        | while IFS= read -r p; do add_violation execs "$(lsmode_of "$p") $p"; done
}

check_symlinks() {
    local link target
    find "$TREE" -type l 2>>"$WORK/errors" | while IFS= read -r link; do
        target=$(resolve_path "$link")
        if [ ! -e "$target" ]; then
            add_violation symlinks "$link -> $target (broken)"
            continue
        fi
        if [ -d "$target" ]; then
            if ! has_o_r "$target" || ! has_o_x "$target"; then
                add_violation symlinks "$link -> $(lsmode_of "$target") $target (target dir not o+rx)"
            fi
        else
            if ! has_o_r "$target"; then
                add_violation symlinks "$link -> $(lsmode_of "$target") $target (target not o+r)"
            fi
        fi
        # A target outside the tree needs its own traversable ancestry.
        case "$target" in
            "$TREE"/*) : ;;
            *)  local d seen
                d=$(dirname "$target")
                seen="$WORK/seen_prefixes"
                while [ "$d" != "/" ]; do
                    grep -qxF "$d" "$seen" 2>/dev/null && break
                    printf '%s\n' "$d" >> "$seen"
                    if ! has_o_x "$d"; then
                        add_violation symlinks "$link -> $target ($(lsmode_of "$d") $d not o+x on target's path)"
                    fi
                    d=$(dirname "$d")
                done ;;
        esac
    done
}

check_acls() {
    if ! command -v getfacl >/dev/null 2>&1; then
        echo "note: getfacl not available — skipping ACL scan." >&2
        return 0
    fi
    echo "note: scanning ACLs (slow on large trees; --no-acl to skip)." >&2
    echo "note: GPFS NFSv4 ACLs do not appear in getfacl output;" \
         "spot-check with mmgetacl if the filesystem uses them." >&2
    # -s: only files with entries beyond the base mode bits; -p: absolute paths.
    getfacl -R -s -p "$TREE" 2>>"$WORK/errors" | awk -v expect="$EXPECT_GROUP" '
        /^# file:/ { path = substr($0, 9); next }
        /^user:[^:]/ {                      # named-user entry: per-uid grant/deny
            print "acl-user\t" path ": " $0; next
        }
        /#effective:/ {                     # mask clamps a granted entry
            print "acl-mask\t" path ": " $0; next
        }
        /^group:[^:]/ {                     # named-group entry other than expected
            split($0, a, ":")
            if (a[2] != expect) print "acl-group\t" path ": " $0
            next
        }
    ' | while IFS=$(printf '\t') read -r category detail; do
        add_violation "$category" "$detail"
    done
}

run_checks() {
    FAILED=0

    if [ -z "$TREE" ]; then
        echo "Error: no tree path given (see --help)." >&2
        exit 2
    fi
    if [ ! -d "$TREE" ]; then
        echo "Error: not a directory: $TREE" >&2
        exit 2
    fi
    # Physical path (-P) so in-tree symlink targets, which resolve physically,
    # compare correctly against the tree prefix.
    TREE=$(cd "$TREE" && pwd -P)

    echo "Checking world-readability of: $TREE"

    [ "$DO_PREFIX" -eq 1 ] && check_prefix
    check_tree_modes
    check_symlinks
    [ "$DO_ACL" -eq 1 ] && check_acls

    report_category prefix \
        "ancestor directory not world-traversable (o+x missing)" \
        "chmod o+x <dir> — if you do not own it (e.g. the project parent on CFS), this is a facilities ticket, not a chmod"
    report_category dirs \
        "directory in tree missing o+rx" \
        "chmod o+rx <dir>   (bulk: setfacl -R -m o::r-X $TREE && setfacl -R -dm o::r-X $TREE, or rootstock setup-perms --retrofit)"
    report_category files \
        "file missing o+r" \
        "chmod o+r <file>   (bulk: as above; capital X avoids marking data files executable)"
    report_category execs \
        "owner-executable file missing o+x" \
        "chmod o+x <file>"
    report_category symlinks \
        "symlink target unreachable for other users" \
        "fix the target's mode/ancestry (listed per link above)"
    report_category acl-user \
        "per-user ACL entry (invisible in ls -l; can grant or deny a specific uid)" \
        "setfacl -x u:<user> <path> after confirming it is not intentional"
    report_category acl-mask \
        "ACL mask clamps a granted entry (classic too-restrictive-umask artifact)" \
        "setfacl -m m::rwX <path>"
    report_category acl-group \
        "unexpected named-group ACL entry" \
        "review; pass --expect-group <g> if this group is intentional"

    if [ -s "$WORK/errors" ]; then
        echo
        echo "warning: some paths could not be inspected:" >&2
        sed 's/^/  /' "$WORK/errors" >&2
    fi

    if [ "$FAILED" -eq 1 ]; then
        echo
        echo "RESULT: FAIL — $TREE does not satisfy the world-readable contract."
        return 1
    fi
    echo "RESULT: OK — $TREE is world-readable (mode bits, symlinks$( [ "$DO_ACL" -eq 1 ] && echo ", ACLs" ))."
    return 0
}

# ---------------------------------------------------------------------------
# Self-test: build a broken fixture, assert every check fires
# ---------------------------------------------------------------------------

self_test() {
    local fixture out rc failures=0
    fixture=$(mktemp -d "${TMPDIR:-/tmp}/cwr_fixture.XXXXXX") || exit 2
    # shellcheck disable=SC2064
    trap "rm -rf '$fixture' '$WORK'" EXIT

    expect() { # expect "label" "pattern" <<< output
        local label="$1" pattern="$2"
        if grep -q "$pattern" <<< "$out"; then
            echo "  ok: detects $label"
        else
            echo "  MISSING: did not detect $label (pattern: $pattern)" >&2
            failures=$((failures + 1))
        fi
    }

    echo "== self-test: clean tree passes =="
    mkdir -p "$fixture/good/sub"
    echo data > "$fixture/good/sub/file"
    echo bin > "$fixture/good/sub/tool"
    ln -s file "$fixture/good/sub/link"
    chmod 755 "$fixture/good" "$fixture/good/sub" "$fixture/good/sub/tool"
    chmod 644 "$fixture/good/sub/file"
    out=$("$0" "$fixture/good" --no-prefix --no-acl 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
        echo "  ok: clean tree exits 0"
    else
        echo "  FAILED: clean tree exited $rc:" >&2
        printf '%s\n' "$out" >&2
        failures=$((failures + 1))
    fi

    echo "== self-test: broken tree fails with each violation reported =="
    mkdir -p "$fixture/bad/dir_no_ox"
    echo secret > "$fixture/bad/file_600"
    echo bin > "$fixture/bad/exec_744"
    echo hidden > "$fixture/bad/target_600"
    ln -s target_600 "$fixture/bad/link_to_600"
    ln -s does-not-exist "$fixture/bad/broken_link"
    chmod 755 "$fixture/bad"
    chmod 750 "$fixture/bad/dir_no_ox"
    chmod 600 "$fixture/bad/file_600" "$fixture/bad/target_600"
    chmod 744 "$fixture/bad/exec_744"
    out=$("$0" "$fixture/bad" --no-prefix --no-acl 2>&1); rc=$?
    if [ $rc -eq 1 ]; then
        echo "  ok: broken tree exits 1"
    else
        echo "  FAILED: broken tree exited $rc (expected 1)" >&2
        failures=$((failures + 1))
    fi
    expect "dir missing o+rx"        "dir_no_ox"
    expect "file missing o+r"        "file_600"
    expect "exec missing o+x"        "exec_744"
    expect "symlink to unreadable"   "link_to_600 ->"
    expect "broken symlink"          "broken)"

    echo "== self-test: non-traversable ancestor detected =="
    mkdir -p "$fixture/wrap/tree"
    echo data > "$fixture/wrap/tree/file"
    chmod 750 "$fixture/wrap"
    chmod 755 "$fixture/wrap/tree"
    chmod 644 "$fixture/wrap/tree/file"
    out=$("$0" "$fixture/wrap/tree" --no-acl 2>&1); rc=$?
    if [ $rc -eq 1 ]; then
        echo "  ok: prefix violation exits 1"
    else
        echo "  FAILED: prefix case exited $rc (expected 1)" >&2
        failures=$((failures + 1))
    fi
    expect "ancestor missing o+x" "\[prefix\]"

    echo
    if [ "$failures" -eq 0 ]; then
        echo "SELF-TEST PASS"
        return 0
    fi
    echo "SELF-TEST FAIL ($failures problem(s))" >&2
    return 1
}

if [ "$SELF_TEST" -eq 1 ]; then
    self_test
    exit $?
fi

run_checks
exit $?
