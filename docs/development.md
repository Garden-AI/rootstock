# Development

## Local development setup

```bash
git clone https://github.com/Garden-AI/rootstock.git
cd rootstock
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Code quality

Run linting before committing:

```bash
ruff check rootstock/
ruff format rootstock/
```

## Project structure

```
rootstock/
├── rootstock/
│   ├── __init__.py
│   ├── calculator.py             # ASE Calculator interface
│   ├── server.py                 # Spawns worker subprocess, manages socket lifecycle
│   ├── worker.py                 # i-PI client state machine
│   ├── environment.py            # Pre-built environment management
│   ├── clusters.py               # Cluster registry and known environments
│   ├── cli.py                    # CLI entry point
│   └── commands/                 # One module per CLI command (install, add, status, ...)
├── sample_model_configurations/  # Sample environment files, grouped by hardware target
├── skill/                        # Agent skill (skill.md)
├── lammps/                       # LAMMPS fix source files
├── tests/
└── docs/
```

## Running tests

```bash
pytest tests/
```

## Get involved

We welcome feedback, bug reports, and collaborators.

For bugs, feature requests, or to contribute an environment file for a new MLIP, [open an issue on GitHub](https://github.com/Garden-AI/rootstock/issues/new).

If you are an HPC admin who would like to support Rootstock on their own cluster or want to discuss a scientific use case, please reach out to Will Engler at [willengler@uchicago.edu](mailto:willengler@uchicago.edu).
