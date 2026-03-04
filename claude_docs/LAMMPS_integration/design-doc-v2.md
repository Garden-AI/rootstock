# Rootstock LAMMPS Integration: Chunk 2 Design Doc

## Overview

Chunk 1 validated that a C++ LAMMPS `fix` can act as an i-PI server and communicate correctly with the unmodified Rootstock Python worker. All three acceptance tests passed: protocol handshake, numerical force agreement (max diff < 1e-6 eV/Å), and 100-step NVE trajectory stability.

Chunk 2 upgrades the fix from a manual two-process workflow into a self-contained, production-ready LAMMPS integration. The user should be able to write a single LAMMPS input script and run it — no separate `rootstock serve` step.

**Scope**: Auto-spawning the worker, `cluster`/`model`/`device` keywords, virial support for NPT, element count validation, and the `rootstock resolve` CLI helper.

**Out of scope**: Multi-node domain decomposition, advanced features like on-the-fly model switching.

---

## Background: What Chunk 1 Delivered

The current `fix_rootstock` requires the user to manually start a worker in a separate terminal:

```bash
# Terminal 1: start LAMMPS (blocks waiting for worker)
mpirun -np 1 lmp_gpu -in script.lammps

# Terminal 2: start worker manually
rootstock serve mace --checkpoint medium --device cpu \
    --socket /path/to/sock --root /scratch/gpfs/.../rootstock
```

This is fine for development but unacceptable for production. Users expect to write a LAMMPS input script and run it.

### Bugs Fixed After Chunk 1 Testing

Two bugs were found and fixed during on-cluster testing:

1. **`FixConst::POST_FORCE`**: Recent LAMMPS (2021+) moved fix constants into the `FixConst` namespace. `setmask()` must return `FixConst::POST_FORCE`, not bare `POST_FORCE`.

2. **Missing `setup()` override**: Without `setup(int vflag)`, the fix contributes no forces on `run 0` or the first timestep of any run. LAMMPS computes initial forces via `setup()`, not `post_force()`. The fix is:
   ```cpp
   void FixRootstock::setup(int vflag) { post_force(vflag); }
   ```

Both are committed and pushed.

---

## 1. Target User Experience

### Syntax

```
fix mlip all rootstock cluster della model mace checkpoint medium device cuda elements Cu O C
```

This single line:
1. Calls `rootstock resolve --cluster della --json` to find the root directory
2. Spawns a worker subprocess using `rootstock serve`
3. Connects via Unix socket (auto-generated path)
4. Exchanges forces every timestep
5. Cleans up the worker on exit

---

## 2. `rootstock resolve` CLI Command

A new CLI command that outputs cluster configuration as JSON. This keeps all cluster registry logic in Python and avoids duplicating it in C++.

### Usage

```bash
rootstock resolve --cluster della --json
```

### Output

```json
{
  "root": "/scratch/gpfs/ROSENGROUP/common/rootstock",
  "cluster": "della"
}
```

### Implementation

Add to `rootstock/cli.py`:

```python
def cmd_resolve(args) -> int:
    """Resolve cluster configuration and print as JSON."""
    import json
    from .clusters import get_root_for_cluster

    try:
        root = get_root_for_cluster(args.cluster)
    except KeyError:
        print(f"Error: unknown cluster '{args.cluster}'", file=sys.stderr)
        return 1

    result = {
        "root": str(root),
        "cluster": args.cluster,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"Cluster: {args.cluster}")
        print(f"Root:    {root}")
    return 0
```

Add subparser:

```python
resolve_parser = subparsers.add_parser(
    "resolve",
    help="Resolve cluster configuration",
)
resolve_parser.add_argument("--cluster", required=True, help="Cluster name")
resolve_parser.add_argument("--json", action="store_true", help="Output as JSON")
resolve_parser.set_defaults(func=cmd_resolve)
```

---

## 3. Auto-Spawn Implementation in C++

### Constructor Changes

The constructor parses keyword arguments starting from `arg[3]`:

```
fix <id> <group> rootstock cluster <name> model <model> checkpoint <ckpt> device <dev> elements <e1> ...
```

All keywords except `elements` and `cluster` have defaults:
- `model`: required (no default)
- `checkpoint`: `"default"` 
- `device`: `"cuda"`
- `timeout`: `120` (seconds, to accommodate CPU model loading)

Parse logic:

```cpp
// In constructor, parse keyword arguments:
std::string cluster_name, model, checkpoint = "default", device = "cuda";
int timeout = 120;

int iarg = 3;  // first keyword after fix-style name
while (iarg < narg) {
    std::string key = arg[iarg];
    if (key == "cluster" && iarg + 1 < narg) { cluster_name = arg[++iarg]; }
    else if (key == "model" && iarg + 1 < narg) { model = arg[++iarg]; }
    else if (key == "checkpoint" && iarg + 1 < narg) { checkpoint = arg[++iarg]; }
    else if (key == "device" && iarg + 1 < narg) { device = arg[++iarg]; }
    else if (key == "timeout" && iarg + 1 < narg) { timeout = std::stoi(arg[++iarg]); }
    else if (key == "elements") { /* parse remaining args as elements */ break; }
    else { error->all(FLERR, "fix rootstock: unknown keyword '{}'", key); }
    iarg++;
}
```

### Spawning the Worker in `init()`

After creating the socket and before accepting a connection:

```cpp
// 1. Resolve cluster root via rootstock CLI
std::string cmd = "rootstock resolve --cluster " + cluster_name + " --json";
FILE *pipe = popen(cmd.c_str(), "r");
// Read JSON output, parse "root" field
// (Use simple string search — no JSON library needed for this minimal format)

// 2. Generate a unique socket path
// Use /tmp/rootstock_<pid>_<fixid>.sock
socket_path_ = "/tmp/rootstock_" + std::to_string(getpid()) + "_" + id + ".sock";

// 3. Create and bind the socket (existing code)
// ...

// 4. Spawn the worker
std::string serve_cmd = "rootstock serve " + model +
    " --root " + root +
    " --socket " + socket_path_ +
    " --checkpoint " + checkpoint +
    " --device " + device +
    " &";  // background
worker_pid_ = fork_and_exec(serve_cmd);
// Or use popen / system() with & for simplicity

// 5. Accept connection with configurable timeout (existing code, use timeout_)
```

### Process Management

The fix must track the worker PID for cleanup:

```cpp
// New private member:
pid_t worker_pid_;

// In destructor, after sending EXIT:
if (worker_pid_ > 0) {
    kill(worker_pid_, SIGTERM);
    int status;
    waitpid(worker_pid_, &status, WNOHANG);
    // If still alive after 5s, SIGKILL
}
```

### Important: `rootstock serve` Must Be on PATH

The worker environment's Python is managed by rootstock, but the `rootstock` CLI itself must be available in the LAMMPS process's `PATH`. This is typically satisfied because the user has rootstock installed in their active Python environment. If not, the fix should produce a clear error:

```
ERROR: fix rootstock: 'rootstock' command not found. 
Install rootstock in your Python environment: pip install rootstock
```

### Important: Fork Safety

`fork()` inside an MPI process can be dangerous on some systems. However, this is standard practice in LAMMPS — `fix external` and `fix python` both spawn child processes. The worker is a completely independent Python process, not an MPI child.

Use `fork()` + `execvp()` rather than `system()` for better signal handling and PID tracking. The fix keywords map 1:1 to `rootstock serve` arguments: `model` becomes the positional arg, `checkpoint` becomes `--checkpoint`:

```cpp
pid_t FixRootstock::spawn_worker(const std::string &root, const std::string &model,
                                  const std::string &checkpoint, const std::string &device)
{
    pid_t pid = fork();
    if (pid < 0) error->all(FLERR, "fix rootstock: fork() failed");
    if (pid == 0) {
        // Child: exec rootstock serve
        execlp("rootstock", "rootstock", "serve",
               model.c_str(),
               "--root", root.c_str(),
               "--socket", socket_path_.c_str(),
               "--checkpoint", checkpoint.c_str(),
               "--device", device.c_str(),
               nullptr);
        // If exec fails:
        _exit(127);
    }
    return pid;  // Parent: return child PID
}
```

---

## 4. Virial / NPT Support

### Current State

The worker already computes and sends the virial tensor. The Chunk 1 fix receives it correctly but discards it:

```cpp
// In recv_forceready():
double virial_au[9];
recvall(virial_au, sizeof(virial_au));
// Virial received but not used
```

### What Needs to Change

LAMMPS fixes contribute to the global virial via the `virial[]` array (6-element Voigt notation). The fix must:

1. **Store the virial** in the class member `virial[6]`
2. **Set `virial_global_flag`** in the constructor
3. **Convert from 3×3 to Voigt** notation

#### Constructor additions:

```cpp
// Enable virial contribution
virial_global_flag = 1;
thermo_virial = 1;
```

#### In `recv_forceready()`:

```cpp
// Virial: 3x3 transposed in Hartree -> Voigt in eV
// i-PI sends column-major 3x3, so after receiving as row-major we have the transpose
double virial_au[9];  // row-major storage of the transposed virial
recvall(virial_au, sizeof(virial_au));

// Un-transpose: virial_ij = virial_au[j*3+i] (since we received the transpose)
// Then convert Hartree -> eV
// Voigt: xx, yy, zz, yz, xz, xy
double h2ev = HARTREE_TO_EV;
virial[0] = virial_au[0] * h2ev;  // xx
virial[1] = virial_au[4] * h2ev;  // yy
virial[2] = virial_au[8] * h2ev;  // zz
virial[3] = virial_au[5] * h2ev;  // yz  (or [7], they should be equal for symmetric virial)
virial[4] = virial_au[2] * h2ev;  // xz  (or [6])
virial[5] = virial_au[1] * h2ev;  // xy  (or [3])
```

#### Sign convention

**Critical**: Verify the sign convention between Rootstock's virial and LAMMPS's virial. ASE uses the stress convention where `virial = -stress * volume`. LAMMPS fixes contribute virial with the sign such that `virial[i]` represents the contribution to `-P*V`. Check by running a short NPT simulation and confirming that the pressure matches what ASE reports for the same configuration.

#### Validation

The virial test should compare the pressure reported by LAMMPS (via `thermo_style ... press`) against the pressure computed by ASE for the same structure:

```python
from ase.build import bulk
atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 2, 2)
# Displace atoms
stress = atoms.get_stress(voigt=True)  # eV/Å³
volume = atoms.get_volume()
pressure_eV_A3 = -np.mean(stress[:3])
pressure_GPa = pressure_eV_A3 * 160.21766208  # eV/Å³ -> GPa
pressure_bar = pressure_GPa * 1e4
```

Compare against LAMMPS `press` output (which is in bar for `units metal`).

---

## 5. Multi-Element Type Mapping Validation

### Current Behavior

The Chunk 1 constructor already validates that the number of elements matches the number of atom types:

```cpp
if (nelem != ntypes)
    error->all(FLERR, "fix rootstock: {} elements given but {} atom types defined", nelem, ntypes);
```

### Enhancement: Warning for Unmapped Types

Add a warning (not error) if some atom types in the simulation have zero atoms. This catches cases where a data file defines more types than are actually used:

```cpp
// In init(), after building atomic_numbers_:
for (int t = 1; t <= atom->ntypes; t++) {
    int count = 0;
    for (int i = 0; i < atom->nlocal; i++)
        if (atom->type[i] == t) count++;
    if (count == 0)
        error->warning(FLERR, "fix rootstock: atom type {} ({}) has no atoms", t, elements_[t-1]);
}
```

---

## 6. Test Procedure

**Important**: Tests are run on Della by Will. Claude Code does not have access to Della.

### Test Environment

- Compile on `della-gpu` login node with `module load openmpi/gcc/4.1.6 cudatoolkit/12.9`
- Run on `--constraint="intel&gpu40"` for GPU tests, or `--partition=mig` with `device cpu` for quick iteration
- LAMMPS requires `mpirun -np 1` to launch (MPI build)
- Use `cubic=True` in `ase.build.bulk()` for orthogonal boxes
- Add `mass` lines to LAMMPS input scripts (ASE's lammps-data writer omits masses)
- Use displaced atoms for force comparisons (perfect crystals have zero forces by symmetry)
- Store test artifacts under `/home/ew2876/rootstock/tests/lammps/` (not `/tmp`, which is cleared between allocations)
- Note: MIG 1g.10gb partitions cause cuBLAS errors with MACE. Use `device cpu` on MIG or `gpu40` for GPU testing.

### Test 1: Auto-Spawn Smoke Test

```
# test_autospawn.lammps
units metal
boundary p p p
lattice fcc 3.615
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 63.546

pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock cluster della model mace checkpoint medium device cpu elements Cu

thermo_style custom step temp pe f_mlip
thermo 1

run 0
```

**Pass criteria**: LAMMPS spawns the worker automatically, completes `run 0`, and cleans up. No manual `rootstock serve` needed. `f_mlip` shows nonzero energy.

### Test 2: Auto-Spawn Force Correctness

Same as Chunk 1 Test 3 but using auto-spawn syntax:

```
fix mlip all rootstock cluster della model mace checkpoint medium device cpu elements Cu
```

Compare forces against ASE reference. Max difference should be < 1e-6 eV/Å (identical to Chunk 1 since the worker is unchanged).

### Test 3: `rootstock resolve` CLI

```bash
rootstock resolve --cluster della --json
# Should output: {"root": "/scratch/gpfs/ROSENGROUP/common/rootstock", "cluster": "della"}
```

### Test 4: Virial / Pressure

Compare pressure from LAMMPS against ASE for the same displaced structure.

```python
# reference_pressure.py
from ase.build import bulk
from rootstock import RootstockCalculator
import numpy as np

np.random.seed(42)
atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 2, 2)
atoms.positions += np.random.normal(0, 0.05, atoms.positions.shape)

with RootstockCalculator(cluster="della", model="mace", checkpoint="medium", device="cpu") as calc:
    atoms.calc = calc
    stress = atoms.get_stress(voigt=True)

volume = atoms.get_volume()
pressure_eV_A3 = -np.mean(stress[:3])
pressure_bar = pressure_eV_A3 * 160.21766208 * 1e4
print(f"ASE pressure: {pressure_bar:.2f} bar")
```

```
# test_virial.lammps
units metal
boundary p p p
read_data reference_structure.data
mass 1 63.546

pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock cluster della model mace checkpoint medium device cpu elements Cu

thermo_style custom step press
thermo 1

run 0
```

**Pass criteria**: LAMMPS `press` matches ASE pressure to within 1%.

### Test 5: NPT Stability

```
# test_npt.lammps
units metal
boundary p p p
read_data reference_structure.data
mass 1 63.546

pair_style zero 6.0
pair_coeff * *

fix mlip all rootstock cluster della model mace checkpoint medium device cpu elements Cu

velocity all create 300 12345
fix npt_int all npt temp 300 300 0.1 iso 0 0 1.0
timestep 0.001

thermo_style custom step temp pe ke press vol f_mlip
thermo 10

run 200
```

**Pass criteria**: 200-step NPT trajectory completes without crashes, hangs, or NaN. Volume fluctuates (confirming the barostat is receiving virial information).

---

## 7. Implementation Checklist

### C++ (fix_rootstock)

- [ ] Parse keyword arguments: `cluster`, `model`, `checkpoint`, `device`, `timeout`, `elements`
- [ ] Implement `spawn_worker()` using `fork()` + `execlp()`
- [ ] Call `rootstock resolve --cluster <name> --json` and parse output
- [ ] Generate unique socket path: `/tmp/rootstock_<pid>_<fixid>.sock`
- [ ] Set accept timeout from `timeout` keyword (default 120s)
- [ ] Track `worker_pid_` for cleanup
- [ ] Clean up worker process in destructor (`SIGTERM` → wait → `SIGKILL`)
- [ ] Store virial in Voigt notation from received 3×3 tensor
- [ ] Set `virial_global_flag = 1` and `thermo_virial = 1` in constructor
- [ ] Verify virial sign convention against ASE
- [ ] Add warning for atom types with zero atoms

### Python (rootstock resolve)

- [ ] Add `cmd_resolve()` to `rootstock/cli.py`
- [ ] Add `resolve` subcommand to argparse in `main()`
- [ ] Support `--json` flag for machine-readable output
- [ ] Return exit code 1 for unknown clusters

### Testing

- [ ] Auto-spawn smoke test (test 1)
- [ ] Auto-spawn force correctness (test 2)
- [ ] `rootstock resolve` CLI (test 3)
- [ ] Virial / pressure comparison (test 4)
- [ ] NPT trajectory stability (test 5)

---

## 8. Open Questions

1. **Virial sign convention**: Need to verify empirically whether the Rootstock virial needs a sign flip to match LAMMPS conventions. The cleanest approach is a pressure comparison test.

2. **`rootstock` on PATH**: The auto-spawn assumes `rootstock` is on PATH. If users install rootstock in a conda env that isn't activated when they run LAMMPS, the spawn will fail. Should we support an explicit `rootstock_path` keyword as fallback?

3. **Worker failure detection**: If the worker crashes mid-simulation (e.g., GPU OOM), the fix will get a broken pipe on the next `send`. Currently this produces a LAMMPS error. Should we attempt restart logic, or is fail-fast the right behavior for Chunk 2?

4. **Socket path for shared nodes**: `/tmp/rootstock_<pid>_<fixid>.sock` should be unique, but on shared nodes with many users, `/tmp` can fill up. Consider using a path under the rootstock root or the user's scratch space instead.