# API Reference

## RootstockCalculator

The main interface to Rootstock is the `RootstockCalculator` class, an ASE-compatible calculator.

### Basic Usage

```python
from ase.build import bulk
from rootstock import RootstockCalculator

atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

with RootstockCalculator(
    cluster="della",
    model="mace",
    checkpoint="medium",
    device="cuda",
) as calc:
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cluster` | `str` | Yes* | Cluster name (e.g., `"della"`, `"sophia"`) |
| `root` | `str` | Yes* | Custom root path instead of a known cluster |
| `model` | `str` | Yes | MLIP family: `"mace"`, `"chgnet"`, `"uma"`, `"tensornet"` |
| `checkpoint` | `str` | No | Specific model weights (uses environment default if omitted) |
| `device` | `str` | No | `"cuda"` (default) or `"cpu"` |

*Either `cluster` or `root` must be provided, but not both.

### Examples

```python
# Using a known cluster with explicit checkpoint
RootstockCalculator(cluster="della", model="mace", checkpoint="medium")

# Using a known cluster with default checkpoint
RootstockCalculator(cluster="della", model="uma")

# Using a custom root path
RootstockCalculator(root="/scratch/gpfs/specific/install/path/rootstock", model="mace")
```

### Context Manager

`RootstockCalculator` should be used as a context manager to ensure proper cleanup of the worker subprocess:

```python
with RootstockCalculator(...) as calc:
    # Use the calculator
    atoms.calc = calc
    energy = atoms.get_potential_energy()
# Worker process is automatically terminated
```

## Available Models

The set of available models varies by cluster and changes as new environments are added. See the [dashboard](https://garden-ai-prod--rootstock-admin-dashboard.modal.run/) for what is currently deployed on each cluster.

### Model Reference

| Model | Environment | Default Checkpoint | Other Checkpoints |
|-------|-------------|-------------------|-------------------|
| `mace` | mace_env | `medium` | `small`, `large` |
| `chgnet` | chgnet_env | (pretrained) | — |
| `uma` | uma_env | `uma-s-1p1` | — |
| `tensornet` | tensornet_env | `TensorNet-MatPES-PBE-v2025.1-PES` | Other MatGL models |

### Checking Available Models

To see what models are available on your cluster:

```bash
rootstock status --cluster della
```

Or programmatically:

```python
from rootstock import list_environments

envs = list_environments(cluster="della")
for env in envs:
    print(f"{env.name}: {env.checkpoints}")
```
