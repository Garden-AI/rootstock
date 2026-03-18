# Installation

Users install only the lightweight `rootstock` package. The heavy ML dependencies (PyTorch, MACE, FAIRChem, etc.) live in the pre-built environments on the cluster.

## Requirements

- Python 3.10 or later
- Access to a supported HPC cluster (Della, Sophia, or a custom installation)

## Install with pip

```bash
pip install rootstock
```

## Install with uv

```bash
uv pip install rootstock
```

## Verify Installation

After installation, verify that Rootstock is working:

```python
from rootstock import RootstockCalculator

# Check available clusters
print(RootstockCalculator.list_clusters())
```

## What Gets Installed

The `rootstock` package is intentionally minimal. It includes:

- The `RootstockCalculator` ASE-compatible calculator
- The i-PI protocol client/server implementation
- CLI tools for cluster administrators

The heavy dependencies (PyTorch, CUDA, MACE, CHGNet, FAIRChem, etc.) are **not** installed on your system. Instead, they live in pre-built virtual environments managed by cluster administrators.
