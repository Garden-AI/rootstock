# Permissions runbook — sophia/polaris and delta

Pre-launch (or periodic) verification that the shared install is usable by any
cluster user. Run on a login node of the target cluster as the maintainer.
Requires rootstock ≥ 0.8 with the `check-perms` subcommand.

These clusters use a single filesystem for install and cache (no split), and
their shared filesystems are Lustre, so `getfacl` sees everything — there is
no GPFS `mmgetacl` step. Sophia and polaris mount the **same Eagle install**:
audit the tree once (from either machine), but do the compute-node check on
both, since they mount Eagle through different fabrics.

For Perlmutter (split filesystems, GPFS install root, purged cache), use
`perlmutter.md` instead.

## 0. Setup

```bash
# sophia/polaris:
#   ROOT=/eagle/Garden-Ai/rootstock    GROUP=<eagle project group>
# delta:
#   ROOT=/work/hdd/data/rootstock      GROUP=<delta project group>
ROOT=changeme
GROUP=changeme
cd ~
curl -sfO https://raw.githubusercontent.com/Garden-AI/rootstock/main/scripts/check_world_readable.sh
chmod +x check_world_readable.sh
```

## 1. Quick check — root + ancestors (seconds)

```bash
rootstock check-perms "$ROOT" --group "$GROUP"
```

Expect `OK: no permission issues found.` (exit 0).

- Ancestor findings are the historically likely failure mode at ALCF (a
  restricted `/eagle` project parent, found 2026-07-06). That is a facilities
  ticket to the directory's owner — no chmod inside the install can fix it.
- Root-level findings → `rootstock setup-perms "$ROOT" --group "$GROUP" --apply`
  (add `--retrofit` if existing files need fixing too).

## 2. Deep audit — full tree (slow: many minutes)

```bash
nohup ./check_world_readable.sh "$ROOT" --expect-group "$GROUP" > audit.log 2>&1 &
```

No output until the end; silence is good.

- `[acl-mask]` findings on plain data files are benign (the mask mirrors a
  non-executable creation mode; masks never affect `other::` access), and a
  `.uv-cache` tree carries them in bulk — only `rootstock install` reads that
  directory. The pathological signature is `#effective:---`, or clamps on
  directories/executables.
- `[dirs]` / `[files]` / `[execs]` / `[symlinks]` / ancestor findings are
  real. Bulk fix (also recalculates clamped masks):

```bash
setfacl -R -m o::r-X "$ROOT" && setfacl -R -dm o::r-X "$ROOT"
```

## 3. Inheritance canary

```bash
canary="$ROOT/.perm-canary-$$"
touch "$canary" && ls -l "$canary" && getfacl -c "$canary"; rm -f "$canary"
```

Want: an inherited `group:$GROUP` entry and `other::r--`. `#effective:rw-` on
a file is healthy. `#effective:---` or `mask::---` means the umask is broken —
fix it (`umask 002`) and re-run the bulk `setfacl` from step 2.

## 4. Cache health

No purge on these roots, but the same idempotent self-heal verifies every
launch checkpoint is present and intact:

```bash
rootstock add --list --root "$ROOT"        # see the launch checkpoint ids
for ckpt in mace-mp-0-medium uma-s-1p1 orb-v2; do   # edit to the launch list
  rootstock add "$ckpt" --no-verify --root "$ROOT"
done
```

## 5. Quota / headroom

```bash
df -h "$ROOT"
lfs quota -gh "$GROUP" "$ROOT" 2>/dev/null   # plus the site quota tool/portal
```

## 6. Compute-node view

Login-node checks don't prove the filesystems mount the same way where jobs
run. On sophia and polaris run this **once per machine** even though the
install is shared. (Interactive jobs: `qsub -I` at ALCF, `srun`/`salloc` on
delta — see site docs for queue names.)

```bash
ls "$ROOT/envs" && head -c 64 "$ROOT/envs/mace/env_source.py" >/dev/null && echo "compute-node read OK"
```
