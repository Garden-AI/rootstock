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

### 1. Start the rootstock worker

In a separate terminal (or as a background job):

```bash
rootstock serve mace --root /scratch/gpfs/SHARED/rootstock \
    --socket /tmp/rootstock_test.sock \
    --model medium \
    --device cuda
```

### 2. Run LAMMPS

```
units           metal
atom_style      atomic
boundary        p p p

read_data       structure.data

fix             mlip all rootstock /tmp/rootstock_test.sock elements Cu

# Energy is available via f_mlip
thermo_style    custom step temp pe f_mlip
thermo          10

run             100
```

### Notes

- The worker must be started **before** LAMMPS runs (the fix waits up to 60 seconds
  for a connection).
- The `elements` keyword maps LAMMPS atom types to element symbols in order
  (type 1 = first element, type 2 = second, etc.).
- Forces are added to existing forces (`+=`), so do not use another pair style
  unless you intend to combine potentials.
- Requires `units metal` (eV, Angstrom, ps).
