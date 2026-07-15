# LAMMPS integration

!!! warning "Experimental"
    LAMMPS support is experimental and far less tested than the ASE path. For most work, use [`RootstockCalculator`](api.md) from Python. If you try the LAMMPS styles and hit issues, reach out.

Rootstock ships two LAMMPS styles, sharing one worker protocol. Both spawn a worker subprocess that runs the MLIP in its pre-built environment and exchange positions/forces over a Unix socket each timestep:

- **`pair_style rootstock`** — the MLIP as a genuine pair style. It contributes to thermo `pe`, `compute pair`, and the pressure natively, and composes with `pair_style hybrid/scaled`. Tools that assume the potential is a pair style (e.g., calphy) require this form. **Use this when the MLIP is the only potential.**
- **`fix rootstock`** — *adds* MLIP forces on top of whatever pair style is defined. Use this to combine an MLIP with another potential; energy is exposed as the fix scalar `f_<id>`.

Both require `units metal` and a single MPI rank (the worker sees all atoms and computes its own neighborhoods; runs on more than one rank are rejected at startup).

## Building the styles

The styles ship as six files (`rootstock_ipi.{h,cpp}`, `fix_rootstock.{h,cpp}`, `pair_rootstock.{h,cpp}`) with no dependencies beyond the C++ standard library and POSIX sockets. Copy them into your LAMMPS `src/` directory and rebuild:

```bash
./lammps/install.sh /path/to/lammps/src
cd /path/to/lammps/build
cmake ../cmake [your usual flags]
make -j 4
```

Rootstock must be installed and on `PATH` so the styles can call `rootstock resolve` and `rootstock serve`:

```bash
pip install rootstock
```

## pair_style rootstock

```lammps
pair_style rootstock cluster <name> checkpoint <ckpt> \
    [device <dev>] [timeout <sec>] [cutoff <r>]
pair_coeff * * <e1> <e2> ...
```

| Keyword | Required | Default | Description |
|---------|----------|---------|-------------|
| `cluster` | yes | — | Cluster name (e.g., `della`) |
| `checkpoint` | yes | — | Canonical checkpoint id (e.g., `mace-mp-0-medium`) |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `timeout` | no | `120` | Seconds to wait for worker startup |
| `cutoff` | no | `1.0` | Neighbor/comm bookkeeping only; the worker builds its own neighborhoods |

The element symbols on `pair_coeff` map atom types in order (type 1 = first element, and so on).

A minimal input fragment:

```lammps
units metal

pair_style rootstock cluster della checkpoint mace-mp-0-medium device cuda
pair_coeff * * Cu

thermo_style custom step temp pe press
```

The MLIP energy appears in `pe` directly — no placeholder pair style, no `f_<id>` bookkeeping. Per-atom energy and stress (`compute pe/atom`, `compute stress/atom`) are **not** provided; the worker reports global quantities only.

## fix rootstock

```lammps
fix <id> <group> rootstock cluster <name> checkpoint <ckpt> \
    device <dev> [timeout <sec>] elements <e1> <e2> ...
```

| Keyword | Required | Default | Description |
|---------|----------|---------|-------------|
| `cluster` | yes | — | Cluster name (e.g., `della`) |
| `checkpoint` | yes | — | Canonical checkpoint id (e.g., `mace-mp-0-medium`) |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `timeout` | no | `120` | Seconds to wait for worker startup |
| `elements` | yes | — | Element symbols mapping atom types (must be last) |

A minimal input fragment (standalone use, with a placeholder pair style):

```lammps
units metal
pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock cluster della checkpoint mace-mp-0-medium device cuda elements Cu

thermo_style custom step temp pe f_mlip press
```

- **Forces are added** (`+=`) to existing forces — that is the point of the fix form; don't combine with a real pair style unless you intend to sum potentials.
- **Energy** is exposed as `f_<id>` (`variable e equal f_mlip` to capture it). `fix_modify <id> energy yes` folds it into `pe`.
- **Ordering matters with force-rescaling fixes** (e.g. `fix ti/spring`): define `fix rootstock` first, so its forces are present in the array when the rescaling fix runs.

## Notes

- `checkpoint` takes the same canonical id as [`RootstockCalculator`](api.md) and the CLI. Run `rootstock list --root <root>` to see what is registered on a cluster.
- **Barostats** (`npt`, `nph`) work with both styles: the virial is passed through from the worker.
- **Single-node only.** MPI-parallel runs are rejected; launch with `mpirun -np 1` on MPI builds.
