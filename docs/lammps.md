# LAMMPS Integration

!!! warning "Experimental"
    LAMMPS integration is experimental and has not been tested as thoroughly as the ASE integration yet. If you try it and run into issues, please reach out.

Rootstock includes a native LAMMPS `fix` that auto-spawns a worker subprocess, giving LAMMPS users access to any Rootstock-managed MLIP.

## Quick Start

Add one line to your LAMMPS input script:

```lammps
fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu
```

The fix handles worker lifecycle, socket communication, and cleanup automatically. Virial information is passed through, so barostats (`npt`, `nph`) work correctly. Energy is accessible via `f_mlip` in thermo output.

## Building the Fix

The fix ships as two files (`fix_rootstock.h`, `fix_rootstock.cpp`) with no dependencies beyond the C++ standard library and POSIX sockets. Copy them into your LAMMPS `src/` directory and rebuild:

```bash
./lammps/install.sh /path/to/lammps/src
cd /path/to/lammps/build
cmake ../cmake [your usual flags]
make -j 4
```

Rootstock must also be installed and on `PATH` so the fix can call `rootstock resolve` and `rootstock serve`:

```bash
pip install rootstock
```

## Fix Syntax

```lammps
fix <id> <group> rootstock cluster <n> model <model> \
    checkpoint <ckpt> device <dev> [timeout <sec>] elements <e1> <e2> ...
```

### Parameters

| Keyword | Required | Default | Description |
|---------|----------|---------|-------------|
| `cluster` | yes | — | Cluster name (e.g., `della`) |
| `model` | yes | — | MLIP family: `mace`, `chgnet`, `uma`, `tensornet` |
| `checkpoint` | no | `default` | Model weights (e.g., `medium`, `uma-s-1p1`) |
| `device` | no | `cuda` | `cuda` or `cpu` |
| `timeout` | no | `120` | Seconds to wait for worker startup |
| `elements` | yes | — | Element symbols mapping atom types (must be last) |

## Example Input Script

```lammps
units metal
atom_style atomic
boundary p p p

# Create or read your structure
lattice fcc 3.6
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 63.546  # Cu

# Use pair_style zero - the fix provides all forces
pair_style zero 6.0
pair_coeff * *

# Rootstock fix
fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu

# Thermo output - energy accessible via f_mlip
thermo_style custom step temp pe f_mlip press
thermo 100

# Run dynamics
velocity all create 300.0 12345
fix nve all nve
run 1000
```

## Important Notes

- **Requires `units metal`**: The fix checks this at startup and will error if a different unit system is used.
- **Use `pair_style zero`**: The fix provides all interatomic forces, so you need a placeholder pair style.
- **Single-node only**: The worker sees all atoms and computes its own neighborhoods. MPI-parallel LAMMPS runs are not yet supported.
- **Element order matters**: The `elements` keyword must be last, and element symbols must map to LAMMPS atom types in order (type 1 = first element, type 2 = second element, etc.).

## Accessing Energy

The fix stores the potential energy computed by the MLIP. Access it in thermo output:

```lammps
thermo_style custom step temp pe f_mlip press
```

Or use it in a variable:

```lammps
variable mlip_energy equal f_mlip
```

## Barostat Support

Virial information is correctly passed through, so NPT and NPH ensembles work:

```lammps
fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu
fix npt all npt temp 300 300 0.1 iso 1.0 1.0 1.0
```
