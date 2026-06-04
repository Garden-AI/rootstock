# Installation

Users install only the lightweight `rootstock` package. The heavy ML dependencies (PyTorch, MACE, FAIRChem, etc.) live in the pre-built environments on the cluster.

## Requirements

- Python 3.10 or later
- Access to a cluster where Rootstock is deployed, or a custom install root

For current cluster and model coverage, see the [Matter Model Almanac](https://garden-ai.github.io/almanac) and the [deployed clusters](clusters.md) page.

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

print("Rootstock installed successfully")
```

You can also resolve a cluster's install root from the command line:

```bash
rootstock resolve --cluster della --json
```

## What Gets Installed

The `rootstock` package is intentionally minimal. It includes:

- The `RootstockCalculator` ASE-compatible calculator
- The i-PI protocol client/server implementation
- CLI tools for cluster administrators

Heavy dependencies (PyTorch, CUDA, MACE, CHGNet, FAIRChem, etc.) are **not** installed on your system. They live in pre-built virtual environments managed by cluster administrators.
