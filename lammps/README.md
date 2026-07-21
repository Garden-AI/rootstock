# rootstock LAMMPS styles

Two LAMMPS styles that communicate with a rootstock MLIP worker over Unix
sockets using the i-PI protocol:

- **`pair_style rootstock`** — the MLIP as a genuine pair style. Energy in
  thermo `pe`, works with `compute pair` and `pair_style hybrid/scaled`.
  Use this when the MLIP is the only potential, and for pair-style-assuming
  drivers (calphy).
- **`fix rootstock`** — adds MLIP forces on top of an existing pair style.
  Use this to combine potentials. Energy via the fix scalar `f_<id>`.

Both share one protocol implementation (`rootstock_ipi.{h,cpp}`).

## Install

Copy the source files into your LAMMPS source tree:

```bash
./install.sh /path/to/lammps/src
```

## Build

Rebuild LAMMPS with CMake:

```bash
cd /path/to/lammps
mkdir -p build && cd build
cmake ../cmake -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Verify

```bash
./lmp -h | grep -i rootstock
```

You should see `rootstock` listed under both pair styles and fix styles.

## Usage

Both styles spawn the worker themselves via `rootstock serve`, so there is no
separate worker to start. `rootstock` must be installed and on `PATH`.

### pair_style (recommended standalone form)

```
units           metal
atom_style      atomic
boundary        p p p

read_data       structure.data

pair_style      rootstock cluster delta checkpoint mace-mp-0-medium device cuda
pair_coeff      * * Cu

thermo_style    custom step temp pe press
thermo          10

run             100
```

### fix (combine with another potential)

```
pair_style      eam/alloy
pair_coeff      * * Cu.eam.alloy Cu

fix             mlip all rootstock cluster della checkpoint mace-mp-0-medium \
                device cuda elements Cu
```

`checkpoint` is a canonical checkpoint id, the same one used by
`RootstockCalculator` and the CLI. The environment providing it must already be
built on that cluster (`rootstock install`).

### Notes

- Both styles wait up to `timeout` seconds (default 120) for the worker they
  spawned to connect back.
- Element symbols map LAMMPS atom types in order (type 1 = first element,
  type 2 = second, etc.) — via `pair_coeff * *` for the pair style, via the
  trailing `elements` keyword for the fix.
- Fix forces are added to existing forces (`+=`). With force-rescaling fixes
  (e.g. `fix ti/spring`), define `fix rootstock` first.
- Requires `units metal` (eV, Angstrom, ps) and a single MPI rank
  (`mpirun -np 1` on MPI builds); multi-rank runs are rejected at startup.
- The pair style reports global energy/virial only — no per-atom energy or
  stress.
