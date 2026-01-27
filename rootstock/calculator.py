"""
ASE-compatible calculator that delegates to an MLIP worker process.

This is the main user-facing interface for Rootstock.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from .clusters import get_root_for_cluster, parse_model_string
from .server import RootstockServer


class RootstockCalculator(Calculator):
    """
    ASE calculator that runs MLIPs in an isolated subprocess.

    This calculator:
    1. Spawns a worker process that loads the MLIP
    2. Communicates via i-PI protocol over Unix sockets
    3. Keeps the worker alive across calculations (no startup overhead)

    Example (v0.3 API - recommended):
        from ase.build import bulk
        from rootstock import RootstockCalculator

        atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

        # Using cluster name
        with RootstockCalculator(
            cluster="modal",
            model="mace-medium",
            device="cuda",
        ) as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())

        # Using explicit root path
        with RootstockCalculator(
            root="/scratch/gpfs/SHARED/rootstock",
            model="mace-medium",
            device="cuda",
        ) as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())

    Example (power user - direct environment path):
        with RootstockCalculator(
            environment="/custom/path/mace.py",
            model="medium",
            device="cuda",
        ) as calc:
            ...
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str = "mace-medium",
        cluster: str | None = None,
        root: str | Path | None = None,
        device: str = "cuda",
        environment: str | None = None,
        log=None,
        **kwargs,
    ):
        """
        Initialize the Rootstock calculator.

        Args:
            model: Model identifier, e.g. "mace-medium", "chgnet", "mace-/path/to/weights.pt".
                   Format is "{environment}-{model_arg}" or just "{environment}" for defaults.
            cluster: Known cluster name ("modal", "della"). Mutually exclusive with root.
            root: Path to rootstock directory. Mutually exclusive with cluster.
            device: PyTorch device ("cuda", "cuda:0", "cpu")
            environment: Direct path to environment file (power user, overrides model parsing).
                        When specified, the model parameter is passed directly to setup().
            log: Optional file object for logging
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)

        self.device = device
        self.log = log

        # Resolve root directory and environment path
        if environment is not None:
            # Power user mode: direct environment path
            # Root is optional in this mode
            self.root = Path(root) if root else None
            self.environment_path = Path(environment)
            self.model_arg = model  # Pass model string directly to setup()

            if not self.environment_path.exists():
                raise FileNotFoundError(f"Environment file not found: {self.environment_path}")
        else:
            # Standard mode: resolve cluster/root and parse model string
            if cluster is not None and root is not None:
                raise ValueError("Cannot specify both 'cluster' and 'root'")

            if cluster is not None:
                self.root = get_root_for_cluster(cluster)
            elif root is not None:
                self.root = Path(root)
            else:
                raise ValueError("Must specify either 'cluster' or 'root'")

            # Parse model string to get environment name and model arg
            env_name, model_arg = parse_model_string(model)
            # Use {env_name}_env.py to avoid shadowing package names (e.g., mace_env.py)
            self.environment_path = self.root / "environments" / f"{env_name}_env.py"
            self.model_arg = model_arg

            if not self.environment_path.exists():
                env_dir = self.root / "environments"
                if env_dir.exists():
                    available = [p.stem for p in env_dir.glob("*.py")]
                else:
                    available = []
                raise FileNotFoundError(
                    f"Environment '{env_name}' not found at {self.environment_path}. "
                    f"Available: {available}"
                )

        # Generate unique socket name to avoid conflicts
        self._socket_name = f"rootstock_{uuid.uuid4().hex[:8]}"
        self._server: RootstockServer | None = None

    def _ensure_server(self):
        """Start server if not already running."""
        if self._server is None:
            self._server = RootstockServer(
                environment_path=self.environment_path,
                model=self.model_arg,
                device=self.device,
                socket_name=self._socket_name,
                root=self.root,
                log=self.log,
            )
            self._server.start()

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """
        Calculate properties for the given atoms.

        This is called by ASE when properties are requested.
        """
        if properties is None:
            properties = self.implemented_properties

        # Call parent to set self.atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        # Ensure server is running
        self._ensure_server()

        # Get results from worker
        energy, forces, virial = self._server.calculate(
            positions=self.atoms.positions,
            cell=np.array(self.atoms.cell),
            atomic_numbers=self.atoms.numbers,
            pbc=list(self.atoms.pbc),
        )

        # Store results
        self.results["energy"] = energy
        self.results["free_energy"] = energy  # No entropy contribution
        self.results["forces"] = forces

        # Convert virial to stress if cell is 3D
        if self.atoms.cell.rank == 3 and any(self.atoms.pbc):
            volume = self.atoms.get_volume()
            stress_tensor = -virial / volume
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress_tensor)
        else:
            self.results["stress"] = np.zeros(6)

    def close(self):
        """Stop the worker process and clean up."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        # Best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
