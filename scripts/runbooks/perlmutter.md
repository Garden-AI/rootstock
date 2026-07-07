# Permissions runbook — Perlmutter

Pre-launch (or periodic) verification that the shared install is usable by any
cluster user. Run everything on a login node as the maintainer unless a step
says otherwise. Requires rootstock ≥ 0.8 with the `check-perms` subcommand.

Perlmutter is the only split-filesystem cluster: install root on CFS (GPFS),
model-weight cache on PSCRATCH (Lustre, purged by access time).

## 0. Setup

```bash
GROUP=m4845                                        # careful: m, not n
ROOT=/global/cfs/cdirs/m4845/rootstock
CACHE_ROOT=/pscratch/sd/w/wengler/rootstock-cache
cd "$SCRATCH"
curl -sfO https://raw.githubusercontent.com/Garden-AI/rootstock/main/scripts/check_world_readable.sh
chmod +x check_world_readable.sh
```

## 1. Quick check — roots + ancestors (seconds)

```bash
rootstock check-perms "$ROOT" --cache-root "$CACHE_ROOT" --group "$GROUP"
```

Expect `OK: no permission issues found.` (exit 0).

- Root-level findings → `rootstock setup-perms --cluster perlmutter --group m4845 --apply`
  (add `--retrofit` if existing files need fixing too).
- Ancestor findings → the directory's owner runs `chmod o+x`, or it's a NERSC
  facilities ticket. No chmod inside the install can fix an ancestor.

## 2. Deep audit — full tree (slow: many minutes per root)

```bash
nohup ./check_world_readable.sh "$ROOT" --expect-group "$GROUP" > audit-root.log 2>&1 &
nohup ./check_world_readable.sh "$CACHE_ROOT" --expect-group "$GROUP" > audit-cache.log 2>&1 &
```

Running both concurrently is fine (different filesystems). No output until the
end; silence is good.

- Expect `CACHE_ROOT`: PASS.
- Expect `ROOT`: FAIL on `[acl-mask]` findings confined to `.uv-cache` —
  **known-benign, accepted 2026-07-07**. ACL masks never affect `other::`
  (world) access, and only `rootstock install` reads the uv cache. Any *other*
  category — or `acl-mask` on directories/executables outside `.uv-cache` —
  is a real finding.
- `[acl-mask]` showing `rwx → rw-` on plain data files anywhere is also
  benign: the mask mirrors the file's non-executable creation mode. The
  pathological signature is `#effective:---`.
- Bulk fix for real tree-level findings (also recalculates clamped masks):

```bash
setfacl -R -m o::r-X <path> && setfacl -R -dm o::r-X <path>
```

## 3. GPFS native-ACL spot-check (CFS only)

`getfacl` cannot see GPFS NFSv4 ACLs, so the audit above can't either.
`mmgetacl` is not on `PATH`; use the full path:

```bash
/usr/lpp/mmfs/bin/mmgetacl "$ROOT"
/usr/lpp/mmfs/bin/mmgetacl "$ROOT/envs/mace/env_source.py"
```

Expect POSIX-style entries (`user::` / `group::` / `other::`) with `other`
having `r` (and `x` on directories). Entries like `special:owner@` mean an
NFSv4 ACL is present — inspect before proceeding. PSCRATCH is Lustre, where
`getfacl` is authoritative; no equivalent step needed there.

## 4. Inheritance canary

```bash
canary="$ROOT/.perm-canary-$$"
touch "$canary" && ls -l "$canary" && getfacl -c "$canary"; rm -f "$canary"
```

Want: an inherited `group:m4845` entry and `other::r--`. `#effective:rw-` on a
file is healthy (files aren't born executable). `#effective:---` or
`mask::---` means the umask is broken — fix it (`umask 002`) and re-run the
bulk `setfacl` from step 2.

## 5. Cache health vs. the PSCRATCH purge

PSCRATCH purges individual files by access time. Files that are written once
and never re-read (auxiliary blobs in HF snapshots, generated kernel sources)
will always show stale atimes even for actively-used models — that is
expected, not a sign the model is dead.

Do **not** touch/cat files to reset atimes; scripted atime refreshes violate
NERSC purge policy. The policy-clean response is idempotent re-add, which
no-ops on intact weights and re-downloads anything the purge ate:

```bash
rootstock add --list --root "$ROOT"        # see the launch checkpoint ids
for ckpt in mace-mp-0-medium uma-s-1p1 orb-v2; do   # edit to the launch list
  rootstock add "$ckpt" --no-verify --root "$ROOT"
done
```

Optional visibility into what's at risk:

```bash
find "$CACHE_ROOT" -type f -atime +42 2>/dev/null \
  | awk 'NR<=5{print} END{print NR" file(s) not accessed in 6+ weeks"}'
```

## 6. Quota

```bash
showquota            # user home + pscratch
showquota m4845      # CFS project space (where .uv-cache's ~900k inodes live)
```

If `showquota` won't take a project argument, use the Iris web portal. The
plain invocation does *not* include CFS.

## 7. Compute-node view (optional if a non-maintainer end-to-end test is scheduled)

Login-node checks don't prove the filesystems mount the same way where jobs
run:

```bash
salloc -q interactive -C gpu -t 10 -A m4845
# on the node:
ls "$ROOT/envs" && head -c 64 "$ROOT/envs/mace/env_source.py" >/dev/null && echo "compute-node read OK"
```
