# fix rootstock — LAMMPS integration

A LAMMPS fix that communicates with a rootstock MLIP worker over Unix sockets
using the i-PI protocol.

## Install

Copy the fix source files into your LAMMPS source tree:

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
./lmp -h | grep rootstock
```

You should see `rootstock` listed under fix styles.

## Usage

The fix spawns the worker itself via `rootstock serve`, so there is no separate
worker to start. `rootstock` must be installed and on `PATH`.

```
units           metal
atom_style      atomic
boundary        p p p

pair_style      zero 6.0
pair_coeff      * *

read_data       structure.data

fix             mlip all rootstock cluster della checkpoint mace-mp-0-medium \
                device cuda elements Cu

# Energy is available via f_mlip
thermo_style    custom step temp pe f_mlip
thermo          10

run             100
```

`checkpoint` is a canonical checkpoint id, the same one used by
`RootstockCalculator` and the CLI. The environment providing it must already be
built on that cluster (`rootstock install`).

### Notes

- The fix waits up to `timeout` seconds (default 120) for the worker it spawned
  to connect back.
- The `elements` keyword maps LAMMPS atom types to element symbols in order
  (type 1 = first element, type 2 = second, etc.) and must come last.
- Forces are added to existing forces (`+=`), so do not use another pair style
  unless you intend to combine potentials.
- Requires `units metal` (eV, Angstrom, ps).