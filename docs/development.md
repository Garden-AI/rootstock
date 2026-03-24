# Development

## Local Development Setup

```bash
git clone https://github.com/Garden-AI/rootstock.git
cd rootstock
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Code Quality

Run linting before committing:

```bash
ruff check rootstock/
ruff format rootstock/
```

## Project Structure

```
rootstock/
├── rootstock/
│   ├── __init__.py
│   ├── calculator.py      # ASE Calculator interface
│   ├── server.py          # Spawns worker subprocess, manages socket lifecycle
│   ├── worker.py          # i-PI client state machine
│   ├── environment.py     # Pre-built environment management
│   ├── clusters.py        # Cluster registry and known environments
│   └── cli.py             # CLI commands (init, install, status, list, etc.)
├── environments/          # Example environment files
├── lammps/                # LAMMPS fix source files
├── tests/
└── docs/
```

## Running Tests

```bash
pytest tests/
```

## Get Involved

Rootstock is an early-stage project. We welcome feedback, bug reports, and collaborators. If you are interested in:

- Deploying Rootstock on your cluster
- Contributing environment files for new MLIPs
- Using Rootstock for a research project

Contact Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).
