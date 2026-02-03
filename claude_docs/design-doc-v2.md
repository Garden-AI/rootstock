# Rootstock v0.2 Design Document

## Overview

Rootstock provides cached, isolated Python environments for Machine Learning Interatomic Potentials (MLIPs) on HPC clusters. It enables researchers to use MLIP calculators through the ASE interface without managing complex dependency requirements, communicating via the i-PI protocol over Unix sockets.

This document covers the second development pass: defining environment configuration and validating with multiple MLIP environments.

## Goals for This Pass

1. Define the environment file contract (what authors must provide)
2. Implement `rootstock test` and `rootstock register` CLI commands
3. Support configurable root directory for shared environments
4. Validate with 2 MLIP environments on Modal (MACE + CHGNet)

## Non-Goals (Deferred)

- Automatic cluster detection / well-known directory lookup
- Model catalog / registry beyond what's in the environment files
- Multi-GPU coordination
- Princeton cluster deployment (next pass)

---

## Environment File Contract

An environment file is a Python file with PEP 723 metadata that defines a `setup()` function.

```python
# mace.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mace-torch>=0.3.0",
#     "ase>=3.22",
#     "torch>=2.0",
# ]
# ///

def setup(model: str, device: str = "cuda") -> "ase.calculators.Calculator":
    """
    Load an MLIP model and return an ASE-compatible calculator.
    
    Args:
        model: Model identifier (e.g., "mace-mp-0") or path to weights file
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu")
    
    Returns:
        An object implementing at minimum:
          - get_potential_energy(atoms) -> float
          - get_forces(atoms) -> ndarray of shape (n_atoms, 3)
    """
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=model, device=device)
```

### Requirements

- Must have PEP 723 metadata block with dependencies
- Must define `setup(model: str, device: str = "cuda")` at module level
- `setup()` must return an ASE-compatible calculator
- Should not hardcode device selection
- May assume `HF_HOME` environment variable points to shared cache

### Optional

- A `MODELS` list/dict for discoverability (nice-to-have, not enforced)

---

## Directory Structure

```
{root}/
├── environments/
│   ├── mace.py           # Registered environment files
│   └── chgnet.py
└── cache/
    └── huggingface/      # Shared weight cache (HF_HOME)
```

The `root` directory is passed explicitly for now. In a future pass, we'll add cluster detection logic.

Note: Unlike typical uv workflows, we do not pre-create venvs. We rely on `uv run` to create and cache environments on the fly based on PEP 723 metadata. The cache location is controlled via `UV_CACHE_DIR`.

---

## CLI Commands

### `rootstock test <env_file.py> --model <model> [--device <device>] [--root <path>]`

Tests that an environment file works correctly.

**Steps:**
1. Parse PEP 723 metadata from the file
2. Generate a wrapper script (see Worker Spawning Flow below)
3. Set `HF_HOME={root}/cache/huggingface`
4. Run the wrapper with `uv run --with rootstock`
5. The wrapper imports the env file and calls `setup(model, device)`
6. Create a small test system (~10 atoms, e.g., `ase.build.bulk("Cu", "fcc", a=3.6) * (2,2,2)`)
7. Run `get_potential_energy()` and `get_forces()`
8. Report success/failure and basic timing

**Exit codes:**
- 0: Success
- 1: Environment creation failed
- 2: Import/setup failed  
- 3: Calculation failed

### `rootstock register <env_file.py> --root <path>`

Registers a tested environment file to the shared directory.

**Steps:**
1. Verify environment file has valid PEP 723 metadata and `setup()` function
2. Copy `env_file.py` to `{root}/environments/{env_name}.py`
3. Print confirmation

**Note:** Registration does not re-run tests. The expectation is that `rootstock test` was run first ("a driven pile is a proven pile").

### `rootstock list --root <path>`

Lists registered environments.

**Output:**
```
Registered environments in /shared/rootstock:
  mace      /shared/rootstock/environments/mace.py
  chgnet    /shared/rootstock/environments/chgnet.py
```

---

## Python API

```python
from rootstock import RootstockCalculator

calc = RootstockCalculator(
    environment="mace",                    # Name of registered environment
    model="mace-mp-0",                     # Passed to setup()
    device="cuda:0",                       # Passed to setup()
    root="/shared/rootstock",              # Where environments live
)

# Standard ASE usage
atoms = ase.build.bulk("Cu", "fcc") * (10, 10, 10)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# Cleanup (terminates worker subprocess)
calc.close()
```

### For unregistered environments (power user path)

```python
calc = RootstockCalculator(
    environment="/path/to/my_custom_env.py",  # Direct path instead of name
    model="/scratch/user/finetuned.pt",
    device="cuda:0",
    root="/shared/rootstock",                  # Still used for cache
)
```

If `environment` is an absolute path, use it directly rather than looking up in `{root}/environments/`.

---

## Worker Spawning Flow

Rootstock generates a wrapper script at runtime and spawns it via `uv run`. This approach:
- Keeps environment files minimal (no rootstock import or boilerplate required)
- Injects rootstock as a dependency without polluting the environment file
- Mirrors patterns used by similar tools

### Step-by-step flow

**1. Read the environment file and extract PEP 723 metadata**

```python
env_file = "/shared/rootstock/environments/mace.py"
with open(env_file) as f:
    content = f.read()
metadata = parse_pep723(content)  # Simple TOML extraction
```

**2. Generate a wrapper script**

The wrapper includes the same PEP 723 metadata, imports the user's setup function, and starts the worker:

```python
# Generated at runtime, written to a temp file
WRAPPER_SCRIPT = """
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///

import sys
sys.path.insert(0, "/shared/rootstock/environments")

from mace import setup
from rootstock.worker import run_worker

run_worker(
    setup_fn=setup,
    model="mace-mp-0",
    device="cuda:0",
    socket_path="/tmp/rootstock-abc123.sock",
)
"""
```

**3. Spawn subprocess**

```bash
uv run --with /path/to/rootstock /tmp/rootstock_wrapper_xyz.py
```

The rootstock path is discovered at runtime from the calling code:

```python
import rootstock
rootstock_path = Path(rootstock.__file__).parent.parent
```

This ensures the worker uses the exact same rootstock version as the caller—no version drift between development and production, no PyPI lookups required.

- uv creates/caches an environment with the MLIP dependencies
- `--with /path/to/rootstock` injects the local rootstock into that environment
- The wrapper runs, calling `setup()` once, then entering the i-PI worker loop

**4. Communication**

The main process and worker communicate via Unix socket using the i-PI protocol. See next section for protocol details.

---

## i-PI Protocol: Atomic Species via INIT

The v0.1 implementation hardcoded Cu atoms because it skipped proper INIT handling. For real systems, the worker needs to know atomic species.

### Background: How i-PI handles species

The standard i-PI pattern is that both server and client start with matching structure files. The protocol only transmits positions/cell because species are assumed known on both ends. For example, DFTB+ requires a "dummy geometry" file from which it reads species.

However, Rootstock's architecture differs—the worker spawns without prior knowledge of the system. Fortunately, the i-PI protocol includes an `initbytes` field in the INIT message specifically for implementation-defined initialization data. This is the protocol's built-in extension point.

### Protocol flow

1. Server sends `STATUS`
2. Client responds `NEEDINIT`
3. Server sends `INIT` with `bead_index` (int) and `initbytes` (bytes)
4. Client parses `initbytes`, stores species, responds `READY`
5. Subsequent `POSDATA` messages only contain cell + positions

### INIT initbytes content

We use JSON in the `initbytes` field for clarity and debuggability:

```json
{
  "numbers": [29, 29, 29, 29],
  "pbc": [true, true, true]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `numbers` | list[int] | Atomic numbers in order (e.g., 29 for Cu, 6 for C) |
| `pbc` | list[bool] | Periodic boundary conditions [x, y, z] |

### Worker behavior

1. On `INIT`: Parse JSON from `initbytes`, store `numbers` and `pbc`
2. On `POSDATA`: Receive cell (3x3) and positions (Nx3) in Bohr
3. Construct `Atoms` object:
   ```python
   atoms = Atoms(
       numbers=self.numbers,
       positions=positions * BOHR_TO_ANGSTROM,
       cell=cell * BOHR_TO_ANGSTROM,
       pbc=self.pbc,
   )
   ```
4. Compute energy/forces/stress via the calculator

### Why JSON for initbytes?

- Human-readable for debugging
- Extensible (can add fields later without breaking existing workers)
- Trivial to parse in Python
- INIT happens once per calculator lifetime, so efficiency doesn't matter
- Other i-PI clients use various formats in initbytes—there's no standard to follow

---

## Relationship to Groundhog

Rootstock and Groundhog are **separate libraries** that follow **shared conventions**.

### Why not depend on Groundhog?

1. **Thin reuse, heavy dependency**: Groundhog pulls in globus-compute-sdk, proxystore, pydantic, rich, typer. Rootstock needs none of that.

2. **PEP 723 parsing is trivial**: It's a standard format, ~50 lines to implement with error handling. Not worth a dependency.

3. **uv does the real work**: Environment creation, caching, dependency resolution - that's all uv. Both libraries are thin wrappers around `uv run`.

4. **Different execution models**: Groundhog is stateless functions over Globus Compute. Rootstock is persistent subprocesses with i-PI protocol. They don't actually share architecture.

5. **Interop through convention, not coupling**: Both use PEP 723, `UV_CACHE_DIR`, standard HuggingFace cache paths. An environment file written for one could work with the other - not because of shared code, but because of shared standards.

### Shared conventions (for cluster interop)

| Convention | Value |
|------------|-------|
| Environment format | PEP 723 inline metadata |
| Environment management | uv |
| Cache directory | `UV_CACHE_DIR` (respects `SCRATCH`, `TMPDIR` fallbacks) |
| Model weights | `HF_HOME` for HuggingFace-hosted models |

---

## Modal Test Plan

**Goal:** Validate the full flow with 2 different MLIP environments on Modal GPU instances.

### Environment 1: MACE

```python
# mace.py
# /// script
# requires-python = ">=3.10"
# dependencies = ["mace-torch>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///

def setup(model: str, device: str = "cuda"):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=model, device=device)
```

### Environment 2: CHGNet

```python
# chgnet.py
# /// script
# requires-python = ">=3.10"
# dependencies = ["chgnet>=0.3.0", "ase>=3.22", "torch>=2.0"]
# ///

def setup(model: str | None, device: str = "cuda"):
    from chgnet.model import CHGNetCalculator
    if model:
        return CHGNetCalculator(model_path=model, use_device=device)
    return CHGNetCalculator(use_device=device)
```

### Test scenarios

1. `rootstock test mace.py --model mace-mp-0` → creates env, downloads weights, runs sanity calc
2. `rootstock register mace.py` → copies to environments/
3. Same for chgnet
4. Run the existing MD benchmark through `RootstockCalculator` API for both
5. Compare overhead to v0.1 baseline

### Success criteria

- Both environments work through the full test → register → use flow
- Overhead remains <5% for 1000+ atom systems
- Switching between environments is just changing the `environment` parameter

---

## Open Questions for Future Passes

1. **Cluster mapping**: How do we discover the root directory automatically? Web endpoint? Config file shipped with library? Environment variable?

2. **Model catalog**: Should environments declare what models they support? How do users discover available models?

3. **Version management**: What happens when someone updates `mace.py` with new deps? Do we version environments?

4. **Concurrent workers**: Can we run multiple workers (different models) on different GPUs simultaneously?

5. **Stress handling**: Detecting environment from atomic symbols so workers can be auto-selected?

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Environment contract | Single `setup(model, device)` function | Minimal API surface, familiar to ASE users |
| Environment naming | Derived from filename | Simple, predictable, avoids ID schemes |
| Model loading | One model per worker lifetime | Simplicity, avoids state management |
| Device selection | Controlled by Rootstock, passed to setup | Enables future multi-GPU coordination |
| Weight caching | Set `HF_HOME` to shared directory | Leverages existing HF machinery |
| Environment caching | Rely on `uv run` (no pre-created venvs) | uv handles caching automatically |
| Workflow | Test → Register | "Proven pile" principle, catches issues early |
| Root directory | Explicit parameter for now | Defer cluster detection to future pass |
| Groundhog relationship | Separate library, shared conventions | Avoid dependency bloat, interop via standards |