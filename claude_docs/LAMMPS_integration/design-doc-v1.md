# Rootstock LAMMPS Integration: Chunk 1 Design Doc

## Overview

This is the first chunk of work toward LAMMPS integration for Rootstock. The goal is to validate that a C++ LAMMPS `fix` can act as an i-PI server and communicate correctly with the existing, unmodified Rootstock Python worker.

**Scope**: Minimal `fix_rootstock` that connects to a manually pre-started worker, exchanges forces on each timestep, and produces numerically correct results.

**Out of scope (deferred to Chunk 2+)**: Auto-spawning the worker, `rootstock resolve` / `cluster` keyword, virial/NPT support, SLURM integration.

---

## Background: Why This Architecture

LAMMPS users run simulations via `lmp -in script.lammps`, not from Python. They compile custom LAMMPS builds with the specific packages they need. This rules out the Python-driver approach (`fix external` + LAMMPS-as-library) and points toward a native LAMMPS `fix` that users compile into their LAMMPS build.

The key insight is that the Rootstock worker side doesn't change at all. `MLIPWorker` already acts as an i-PI client—it receives positions, computes forces, sends them back. It doesn't know or care who's on the other end. Today that's `RootstockServer` (Python); in this design, it's `fix_rootstock` (C++).

### Architecture

```
LAMMPS process (lmp -in script.lammps)      Rootstock worker process
┌──────────────────────────────────┐        ┌──────────────────────────┐
│  standard LAMMPS input script    │        │  Pre-built venv Python   │
│  ...                             │        │  (mace_env/bin/python)   │
│  fix mlip all rootstock          │        │                          │
│      /tmp/rs.sock elements Cu    │  Unix  │  worker.py (i-PI client) │
│  ...                             │◄──────►│  - receives positions    │
│  run 10000                       │ socket │  - computes MLIP forces  │
│                                  │ (i-PI) │  - sends forces back     │
└──────────────────────────────────┘        └──────────────────────────┘
```

### Why a `fix` (not a `pair_style`)

A `pair_style` operates on atom pairs using LAMMPS neighbor lists. MLIPs don't work that way—they need all atom positions and compute their own neighborhoods internally. A `fix` is the right abstraction because we're injecting total per-atom forces from an external source, exactly like `fix external` and `fix client/md` do.

### Design Limitation: Single-Node Only

By using a `fix` rather than a `pair_style`, we bypass LAMMPS's neighbor list and domain decomposition infrastructure. The MLIP worker sees all atoms and computes its own neighborhoods. This is fine for single-GPU work (Rootstock's current scope). Multi-node domain decomposition would require a fundamentally different approach.

---

## 1. The LAMMPS Fix: `fix_rootstock`

### User-Facing Syntax

```
fix <fix-id> <group> rootstock <socket_path> elements <e1> <e2> ...
```

Example:

```
fix mlip all rootstock /tmp/rootstock_test.sock elements Cu
fix mlip all rootstock /tmp/rootstock_test.sock elements Cu O C
```

The `elements` keyword maps LAMMPS atom types (1, 2, 3...) to chemical elements. This is the standard LAMMPS convention used by `pair_coeff` in DeepMD, MACE, and other MLIP integrations.

### What the Fix Does

On `init()` (LAMMPS setup phase):

1. Parse arguments: extract socket path and element list
2. Convert element symbols to atomic numbers (Cu→29, O→8, C→6) using a lookup table
3. Create a Unix domain socket, listen, and accept a connection from the worker (blocks with timeout)
4. Perform the i-PI initialization handshake (STATUS → NEEDINIT → INIT with atomic numbers JSON)

On `post_force()` (every timestep):

1. Gather all atom positions and the simulation cell from LAMMPS
2. Send STATUS, receive READY
3. Send POSDATA (cell + positions in atomic units)
4. Send STATUS, receive HAVEDATA
5. Send GETFORCE, receive FORCEREADY (energy + forces + virial in atomic units)
6. Convert forces from atomic units to LAMMPS `metal` units (they happen to be the same: eV/Å)
7. Add forces to each atom in the fix group
8. Store energy for thermo output

On cleanup:

1. Send EXIT to the worker
2. Close socket and remove socket file

### Required LAMMPS Includes and Conventions

The fix needs to follow LAMMPS coding conventions:

- Header: `fix_rootstock.h` with the `FixStyle` macro
- Source: `fix_rootstock.cpp`
- Class: `FixRootstock` inheriting from `Fix`
- Must implement: `init()`, `post_force()`, `setmask()`, and constructor/destructor
- `setmask()` returns `POST_FORCE` to tell LAMMPS when to call it
- Must support `fix_modify energy yes` for thermo output

### Units

The fix requires `units metal` (Å, eV, ps, K). This is checked in `init()` and raises an error for any other unit system. This is a non-constraint for MLIP users since metal units are the natural unit system for atomistic simulations with MLIPs.

LAMMPS `metal` units align with ASE units (both use Å and eV), which simplifies conversion. The only conversion needed is between LAMMPS/ASE units and i-PI atomic units (Bohr, Hartree), which the existing protocol code already handles.

Unit conversion constants (matching `rootstock/protocol.py`):

```cpp
constexpr double BOHR_TO_ANGSTROM = 0.52917721067;
constexpr double HARTREE_TO_EV = 27.211386245988;
constexpr double ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM;
constexpr double EV_TO_HARTREE = 1.0 / HARTREE_TO_EV;
```

---

## 2. i-PI Protocol in C++

The protocol is simple: 12-byte ASCII commands and raw binary arrays over a Unix socket. The existing Python implementation in `rootstock/protocol.py` is the reference. The C++ implementation needs these operations:

### Low-Level Operations

```cpp
// Send a 12-byte command string (right-padded with spaces)
void sendmsg(const std::string &msg);

// Receive a 12-byte command string (stripped of trailing spaces)
std::string recvmsg();

// Send/receive raw byte buffers (for arrays)
void sendall(const void *buf, size_t nbytes);
void recvall(void *buf, size_t nbytes);
```

### High-Level Protocol Messages

The server-side protocol sequence for each timestep, matching the flow in `RootstockServer.calculate()`:

**Initialization (first timestep only):**

```
Server sends:  STATUS          (12 bytes)
Worker replies: NEEDINIT       (12 bytes)
Server sends:  INIT            (12 bytes)
                bead_index     (1 × int32 = 4 bytes)
                nbytes         (1 × int32 = 4 bytes)  — length of JSON
                init_bytes     (nbytes × byte)         — JSON payload
Server sends:  STATUS          (12 bytes)
Worker replies: READY          (12 bytes)
```

The `init_bytes` JSON payload (matching `rootstock/server.py`):

```json
{"numbers": [29, 29, 29, 29], "pbc": [true, true, true]}
```

Where `numbers` is the list of atomic numbers for all atoms (in order), and `pbc` indicates periodic boundary conditions per axis.

**Each timestep:**

```
Server sends:  STATUS          (12 bytes)
Worker replies: READY          (12 bytes)
Server sends:  POSDATA         (12 bytes)
                cell           (9 × float64 = 72 bytes)  — 3×3 cell, transposed, in Bohr
                icell          (9 × float64 = 72 bytes)  — 3×3 inverse cell, transposed, in 1/Bohr
                natoms         (1 × int32 = 4 bytes)
                positions      (natoms×3 × float64)       — in Bohr
Server sends:  STATUS          (12 bytes)
Worker replies: HAVEDATA       (12 bytes)
Server sends:  GETFORCE        (12 bytes)
Worker replies: FORCEREADY     (12 bytes)
                energy         (1 × float64 = 8 bytes)    — in Hartree
                natoms         (1 × int32 = 4 bytes)
                forces         (natoms×3 × float64)       — in Hartree/Bohr
                virial         (9 × float64 = 72 bytes)   — 3×3, transposed, in Hartree
                nextra         (1 × int32 = 4 bytes)
                extra          (nextra × byte)
```

After the first call, NEEDINIT is replaced by READY (the worker stays initialized).

### Critical Detail: Cell Transpose Convention

The i-PI protocol transmits the cell matrix **transposed** (column-major). From `rootstock/protocol.py`:

```python
# Sending (server side):
cell_bohr = cell.T * ANGSTROM_TO_BOHR       # transpose then convert
icell_bohr = np.linalg.pinv(cell).T / ANGSTROM_TO_BOHR

# Receiving (server side):
cell_bohr = self.recv_array((3, 3), np.float64).T.copy()  # receive then transpose back
```

The C++ implementation must match this exactly. LAMMPS stores the cell as `boxlo`, `boxhi`, and tilt factors (`xy`, `xz`, `yz`). These need to be assembled into a 3×3 matrix, transposed, and converted to Bohr before sending.

### Critical Detail: Virial Transpose

Same convention applies to the virial tensor:

```python
# Receiving virial (server side):
virial_au = self.recv_array((3, 3), np.float64).T.copy()

# Sending virial (worker side):
self.send_array(virial_au.T, np.float64)
```

For Chunk 1, the virial is received but not used (no NPT support). It should still be received correctly to keep the protocol in sync.

### Byte Order

All data is in the **native byte order** of the machine. Since both LAMMPS and the Rootstock worker run on the same node, this is always consistent. The protocol uses `float64` (IEEE 754 double) and `int32` (32-bit signed integer), matching numpy's defaults.

---

## 3. `rootstock serve` CLI Command

A new CLI command that starts an MLIP worker as a standalone process, connecting to a socket created by an external process (in this case, the LAMMPS fix).

### Usage

```bash
rootstock serve <env_name> --model <model> --device <device> --socket <path> --root <path>
```

Example:

```bash
rootstock serve mace_env --model medium --device cuda --socket /tmp/rootstock_test.sock --root /scratch/gpfs/ROSENGROUP/common/rootstock
```

### How It Works

The roles mirror the existing `RootstockServer`/`MLIPWorker` architecture:

- The **fix** is the i-PI server: it creates the Unix socket, listens, and accepts a connection
- The **worker** (spawned by `rootstock serve`) is the i-PI client: it connects to the socket with retries

`rootstock serve` does the following:

1. Loads the environment config from `{root}/envs/{env_name}/`
2. Generates the wrapper script (existing `EnvironmentManager` logic)
3. Spawns the worker subprocess, telling it to connect to the given socket path
4. Blocks until the worker exits, forwarding SIGTERM/SIGINT for clean shutdown
5. Exits with the worker's exit code

This means **LAMMPS must start first** so the socket exists when the worker tries to connect. The worker's existing retry logic (`connect_unix_socket` with `max_retries=50, retry_delay=0.1`) handles the brief race window.

### Changes to CLI

Add to `rootstock/cli.py`:

```python
# serve command
serve_parser = subparsers.add_parser(
    "serve",
    help="Start a Rootstock worker for external connections",
    description="Start an MLIP worker that connects to a given socket path.",
)
serve_parser.add_argument("env_name", help="Environment name (e.g., mace_env)")
serve_parser.add_argument("--model", required=True, help="Model identifier")
serve_parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
serve_parser.add_argument("--socket", required=True, help="Unix socket path to connect to")
serve_parser.add_argument("--root", required=True, help="Rootstock root directory")
serve_parser.set_defaults(func=cmd_serve)
```

---

## 4. Element Symbol to Atomic Number Mapping

The fix needs a lookup table from element symbols to atomic numbers. This is a static table compiled into the fix. Include the full periodic table (118 entries) to avoid "element not found" surprises.

```cpp
static const std::map<std::string, int> ELEMENT_TO_Z = {
    {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},
    {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
    {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},
    {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20},
    // ... full periodic table through Og (118)
};
```

---

## 5. LAMMPS Build Integration

### File Layout

The fix is distributed as source files that users drop into their LAMMPS build:

```
rootstock-lammps/
├── fix_rootstock.h
├── fix_rootstock.cpp
├── README.md           # Build instructions
└── install.sh          # Optional: copies files to LAMMPS src/
```

### Build Instructions

For users who build LAMMPS with CMake (the recommended method):

```bash
# Copy fix source files into LAMMPS source tree
cp fix_rootstock.h fix_rootstock.cpp /path/to/lammps/src/

# Build LAMMPS as usual (fix is automatically picked up from src/)
cd /path/to/lammps/build
cmake ../cmake [your usual flags]
make -j$(nproc)
```

Files placed directly in `src/` are always compiled — no package enable step needed.

### Dependencies

The fix has **no external dependencies** beyond the C++ standard library and POSIX sockets (for Unix domain sockets). No Boost, no extra libraries, no Python. This is intentional — it keeps the build simple and avoids conflicts with users' existing LAMMPS configurations.

---

## 6. Test Procedure

**Important**: These tests are run manually on Della by Will. Claude Code does not have access to Della.

### Prerequisites

- Rootstock is deployed on Della at `/scratch/gpfs/ROSENGROUP/common/rootstock` with the `mace_env` environment built
- LAMMPS is compiled with `fix_rootstock` on Della (Ryan or Will compiles it)
- A GPU node is available (interactive session or batch job)

### Startup Ordering

The fix creates the socket and listens during `init()`. The worker connects with retries. Therefore **LAMMPS must start first**, and it will block during `init()` waiting for the worker to connect (with a configurable timeout). The test procedure uses two terminals on the same GPU node.

### Test 1: LAMMPS Compiles with the Fix

```bash
# Copy fix files into LAMMPS source
cp fix_rootstock.h fix_rootstock.cpp /path/to/lammps/src/

# Rebuild LAMMPS
cd /path/to/lammps/build
cmake ../cmake -DBUILD_SHARED_LIBS=on [other flags from Ryan's scripts]
make -j$(nproc)

# Verify fix is available
lmp -h | grep rootstock
```

**Pass criteria**: LAMMPS compiles without errors and `rootstock` appears in the fix list.

### Test 2: Protocol Handshake

Run in two terminals on the same GPU node (e.g., via `salloc --gres=gpu:1`):

**Terminal 1 — Start LAMMPS (creates socket and waits for worker):**

```bash
cat > test_rootstock.lammps << 'EOF'
units metal
boundary p p p
lattice fcc 3.615
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 63.546

pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock /tmp/rootstock_test.sock elements Cu

thermo_style custom step temp pe ke etotal press
thermo 1

run 0
EOF

lmp -in test_rootstock.lammps
```

**Terminal 2 — Start the worker (connects to socket):**

```bash
rootstock serve mace_env --model medium --device cuda \
    --socket /tmp/rootstock_test.sock \
    --root /scratch/gpfs/ROSENGROUP/common/rootstock
```

**Pass criteria**: LAMMPS completes `run 0` without errors or timeouts. Worker log shows successful INIT handshake.

### Test 3: Numerical Correctness

Compare forces from LAMMPS+Rootstock against ASE+Rootstock on the same structure.

**Step A — Generate reference forces via ASE:**

```python
# reference_forces.py
from ase.build import bulk
from ase.io import write
from rootstock import RootstockCalculator
import numpy as np

# Create 2x2x2 FCC Cu supercell (32 atoms)
atoms = bulk("Cu", "fcc", a=3.615) * (2, 2, 2)

with RootstockCalculator(
    cluster="della",
    model="mace",
    checkpoint="medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

print(f"Energy: {energy:.10f} eV")
np.savetxt("reference_forces.txt", forces, fmt="%.10f")
write("reference_structure.xyz", atoms)
write("reference_structure.data", atoms, format="lammps-data")
```

**Step B — Run LAMMPS on the same structure:**

```
# test_forces.lammps
units metal
boundary p p p
read_data reference_structure.data

pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock /tmp/rootstock_test.sock elements Cu

# Dump forces after a single evaluation
dump forces all custom 0 lammps_forces.txt id fx fy fz
dump_modify forces sort id
run 0
```

**Step C — Compare:**

```python
import numpy as np

ref = np.loadtxt("reference_forces.txt")
lammps = np.loadtxt("lammps_forces.txt", skiprows=9)[:, 1:]  # skip header, take fx fy fz

max_diff = np.max(np.abs(ref - lammps))
print(f"Max force difference: {max_diff:.2e} eV/Å")
assert max_diff < 1e-6, f"Forces don't match: max diff = {max_diff}"
```

**Pass criteria**: Maximum per-component force difference is < 1e-6 eV/Å. Energy matches to the same tolerance.

### Test 4: Stability Over Trajectory

```
# test_nve.lammps
units metal
boundary p p p
read_data reference_structure.data

pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock /tmp/rootstock_test.sock elements Cu

velocity all create 300 12345

fix nve_int all nve
timestep 0.001

thermo_style custom step temp pe ke etotal press
thermo 10

run 100
```

**Pass criteria**: 100-step NVE trajectory completes without crashes, hangs, or NaN values in thermo output.

---

## 7. Implementation Checklist

### C++ (fix_rootstock)

- [ ] `fix_rootstock.h` — Class declaration with `FixStyle` macro
- [ ] `fix_rootstock.cpp` — Implementation:
  - [ ] Constructor: parse arguments (socket path, element list)
  - [ ] `init()`: validate `units metal`, create socket, listen, accept connection, perform i-PI INIT handshake
  - [ ] `post_force()`: STATUS→READY→POSDATA→STATUS→HAVEDATA→GETFORCE→FORCEREADY cycle
  - [ ] Destructor: send EXIT, close socket, clean up socket file
  - [ ] `setmask()`: return `POST_FORCE`
  - [ ] Energy storage for thermo output
  - [ ] Element symbol → atomic number lookup table (full periodic table)
  - [ ] Socket helper functions (sendall, recvall, sendmsg, recvmsg)
  - [ ] LAMMPS box → 3×3 cell matrix conversion
  - [ ] Unit conversions (Å↔Bohr, eV↔Hartree)

### Python (rootstock serve)

- [ ] Add `cmd_serve()` to `rootstock/cli.py`
- [ ] Add `serve` subcommand to argparse setup in `main()`
- [ ] Reuse `EnvironmentManager` for worker spawning
- [ ] Forward SIGTERM/SIGINT to worker for clean shutdown
- [ ] Block until worker exits, forward exit code

### Build Integration

- [ ] `README.md` with build instructions for CMake
- [ ] Optional `install.sh` script to copy files into LAMMPS src/

### Test Artifacts

- [ ] `reference_forces.py` — Generate reference structure + forces via ASE
- [ ] `test_rootstock.lammps` — Handshake test input
- [ ] `test_forces.lammps` — Force comparison input
- [ ] `test_nve.lammps` — Trajectory stability input
- [ ] `compare_forces.py` — Comparison script

---

## 8. Future Work (Chunk 2+)

Decisions already made but deferred:

- **Auto-spawning**: The fix will spawn the worker itself using `rootstock resolve --cluster della --json` to locate the root directory. This keeps the cluster registry in Python and avoids duplicating it in C++.
- **`cluster` keyword**: `fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu O C` — the full user-facing syntax.
- **Virial/NPT**: The worker already computes the virial. The fix will pass it to LAMMPS via `fix_modify virial yes`.
- **SLURM integration**: Transparent — the fix spawns the worker as a child process which inherits the SLURM GPU environment.
- **Multi-element type mapping validation**: Warn if the number of elements doesn't match the number of LAMMPS atom types.

---

## 9. Key Reference: Existing Protocol Implementation

The C++ implementation must match the Python implementation byte-for-byte. The authoritative reference files are:

- `rootstock/protocol.py` — Low-level i-PI protocol (send/recv messages and arrays, unit conversions)
- `rootstock/server.py` — Server-side state machine (`RootstockServer.calculate()`)
- `rootstock/worker.py` — Client-side state machine (`MLIPWorker.run()`) — the fix must be a compatible peer to this

The protocol state machine from the server's perspective (implemented in `RootstockServer.calculate()`):

```
send STATUS → recv NEEDINIT → send INIT(json) → send STATUS → recv READY
→ send POSDATA(cell, positions) → send STATUS → recv HAVEDATA
→ send GETFORCE → recv FORCEREADY(energy, forces, virial, extra)
```

After the first call, NEEDINIT is replaced by READY (the worker stays initialized).