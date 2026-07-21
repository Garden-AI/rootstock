# Permissions runbook — Perlmutter

Verifies the shared install is usable by any cluster user. Run on a login node
as the maintainer. Install root is on CFS (GPFS); the model-weight cache is on
PSCRATCH (Lustre, purged by access time).

## 0. Setup

```bash
GROUP=m5268                                        # careful: m, not n
ROOT=/global/cfs/cdirs/m5268/rootstock
CACHE_ROOT=/pscratch/sd/o/oprice/rootstock-cache
cd "$SCRATCH"
curl -sfO https://raw.githubusercontent.com/Garden-AI/rootstock/main/scripts/check_world_readable.sh
chmod +x check_world_readable.sh
```

## 1. Quick check — roots + ancestors (seconds)

```bash
rootstock check-perms "$ROOT" --cache-root "$CACHE_ROOT" --group "$GROUP"
```

Expect `OK: no permission issues found.`

- Root-level findings: `rootstock setup-perms --cluster perlmutter --group m5268 --apply` (add `--retrofit` to fix existing files too).
- Ancestor findings: `chmod o+x` by the directory's owner, or a facilities ticket.

## 2. Deep audit — full tree (many minutes per root; no output until the end)

```bash
nohup ./check_world_readable.sh "$ROOT" --expect-group "$GROUP" > audit-root.log 2>&1 &
nohup ./check_world_readable.sh "$CACHE_ROOT" --expect-group "$GROUP" > audit-cache.log 2>&1 &
```

Expect `CACHE_ROOT`: PASS. Expect `ROOT`: FAIL with `[acl-mask]` findings
confined to `.uv-cache` — known-benign (masks never affect `other::`, and only
`rootstock install` reads that directory). `[acl-mask]` on plain data files is
likewise benign; the bad signature is `#effective:---`, or clamps on
directories/executables. Anything in another category is real. Bulk fix, which
also recalculates clamped masks:

```bash
setfacl -R -m o::r-X <path> && setfacl -R -dm o::r-X <path>
```

## 3. GPFS native-ACL spot-check (CFS)

`getfacl` can't see GPFS NFSv4 ACLs; `mmgetacl` isn't on `PATH`:

```bash
/usr/lpp/mmfs/bin/mmgetacl "$ROOT"
/usr/lpp/mmfs/bin/mmgetacl "$ROOT/envs/mace/env_source.py"
```

Expect POSIX-style entries (`user::` / `group::` / `other::`) with `other`
having `r` (and `x` on directories). `special:...@` entries mean an NFSv4 ACL
is present — inspect before proceeding.

## 4. Inheritance canary

```bash
canary="$ROOT/.perm-canary-$$"
touch "$canary" && ls -l "$canary" && getfacl -c "$canary"; rm -f "$canary"
```

Want an inherited `group:m5268` entry and `other::r--`. (`#effective:rw-` on a
file is healthy.) `#effective:---` or `mask::---` means a broken umask — set
`umask 002` and re-run the bulk `setfacl` from step 2.

## 5. Checkpoint presence (PSCRATCH purge recovery)

`add` is idempotent: it no-ops on intact weights and re-downloads anything the
purge removed.

```bash
rootstock add --list --root "$ROOT"                 # the checkpoint ids
for ckpt in mace-mp-0-medium uma-s-1p1 orb-v2; do   # edit to the launch list
  rootstock add "$ckpt" --no-verify --root "$ROOT"
done
```

## 6. Quota

```bash
showquota            # home + pscratch only
showquota m5268      # CFS project space (or the Iris web portal)
```

## 7. Compute-node view

```bash
salloc -q interactive -C gpu -t 10 -A m5268
# on the node:
ls "$ROOT/envs" && head -c 64 "$ROOT/envs/mace/env_source.py" >/dev/null && echo "compute-node read OK"
```
