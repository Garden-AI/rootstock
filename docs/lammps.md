# LAMMPS integration

!!! warning "Experimental"
    LAMMPS support is experimental and far less tested than the ASE path. For most work, use [`RootstockCalculator`](api.md) from Python. If you try the LAMMPS fix and hit issues, reach out.

Rootstock includes a native LAMMPS `fix` that spawns a worker subprocess, giving a LAMMPS run access to a Rootstock-managed MLIP for molecular dynamics. The fix handles worker lifecycle, socket communication, and cleanup. Virial information is passed through, so barostats (`npt`, `nph`) work, and the MLIP energy is available as `f_mlip` in thermo output.

!!! note "Fix keyword interface predates the v0.8 API"
    The fix has not yet been updated to the v0.8 canonical-checkpoint-id API used by the Python interface and CLI. It still takes separate `model` and `checkpoint` keywords, matching the current `lammps/fix_rootstock.cpp` source. The interface is expected to change to a single canonical checkpoint id. The syntax below is what the fix accepts today.

## Building the fix

The fix ships as two files (`fix_rootstock.h`, `fix_rootstock.cpp`) with no dependencies beyond the C++ standard library and POSIX sockets. Copy them into your LAMMPS `src/` directory and rebuild:

```bash
./lammps/install.sh /path/to/lammps/src
cd /path/to/lammps/build
cmake ../cmake [your usual flags]
make -j 4
```

Rootstock must be installed and on `PATH` so the fix can call `rootstock resolve` and `rootstock serve`:

```bash
pip install rootstock
```

## Fix syntax

```lammps
fix <id> <group> rootstock cluster <name> model <model> \
    checkpoint <ckpt> device <dev> [timeout <sec>] elements <e1> <e2> ...
```

| Keyword | Required | Default | Description |
|---------|----------|---------|-------------|
| `cluster` | yes | — | Cluster name (e.g., `della`) |
| `model` | yes | — | MLIP family (e.g., `mace`) |
| `checkpoint` | no | `default` | Model weights within that family (e.g., `medium`) |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `timeout` | no | `120` | Seconds to wait for worker startup |
| `elements` | yes | — | Element symbols mapping atom types (must be last) |

A minimal input fragment:

```lammps
units metal
pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu

thermo_style custom step temp pe f_mlip press
```

## Notes

- **Requires `units metal`.** The fix checks this at startup.
- **Use `pair_style zero`.** The fix provides all interatomic forces, so a placeholder pair style is needed.
- **Single-node only.** The worker sees all atoms and computes its own neighborhoods. MPI-parallel runs are not supported.
- **Element order matters.** `elements` must be last; symbols map to atom types in order (type 1 = first element, and so on).
- **Energy** is exposed as `f_mlip` (`variable e equal f_mlip` to capture it). **Barostats** work because the virial is passed through.
