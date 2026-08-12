# Permissions runbook — sophia/polaris and delta

Verifies the shared install is usable by any cluster user. Run on a login node
as the maintainer. Sophia and polaris mount the same Eagle install: audit the
tree once, but run the compute-node check on both machines.

## 0. Setup

```bash
# sophia/polaris — audit the serving copy users resolve (our mirror under the
# Garden-Ai project; group-scoped, so users need Garden-Ai membership — fix
# findings by re-running scripts/mirror_alcf.sh rather than setup-perms):
#   ROOT=/lus/eagle/projects/Garden-Ai/rootstock    GROUP=<Garden-Ai project group>
# sophia/polaris build root (ours; admin jobs only, not world-readable by design):
#   ROOT=/eagle/projects/Rootstock/rootstock    GROUP=Rootstock
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

Expect `OK: no permission issues found.`

- Root-level findings: `rootstock setup-perms "$ROOT" --group "$GROUP" --apply` (add `--retrofit` to fix existing files too).
- Ancestor findings (e.g. a restricted `/eagle` project parent): `chmod o+x` by the directory's owner, or a facilities ticket.

## 2. Deep audit — full tree (many minutes; no output until the end)

```bash
nohup ./check_world_readable.sh "$ROOT" --expect-group "$GROUP" > audit.log 2>&1 &
```

`[acl-mask]` findings on plain data files are benign (masks never affect
`other::`), and a `.uv-cache` tree carries them in bulk; the bad signature is
`#effective:---`, or clamps on directories/executables. Anything in another
category is real. Bulk fix, which also recalculates clamped masks:

```bash
setfacl -R -m o::r-X "$ROOT" && setfacl -R -dm o::r-X "$ROOT"
```

## 3. Inheritance canary

```bash
canary="$ROOT/.perm-canary-$$"
touch "$canary" && ls -l "$canary" && getfacl -c "$canary"; rm -f "$canary"
```

Want an inherited `group:$GROUP` entry and `other::r--`. (`#effective:rw-` on a
file is healthy.) `#effective:---` or `mask::---` means a broken umask — set
`umask 002` and re-run the bulk `setfacl` from step 2.

## 4. Quota / headroom

```bash
df -h "$ROOT"
lfs quota -gh "$GROUP" "$ROOT" 2>/dev/null   # plus the site quota tool/portal
```

## 5. Compute-node view

Interactive jobs: `qsub -I` at ALCF, `srun`/`salloc` on delta.

```bash
ls "$ROOT/envs" && head -c 64 "$ROOT/envs/mace/env_source.py" >/dev/null && echo "compute-node read OK"
```
