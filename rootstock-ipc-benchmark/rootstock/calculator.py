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

from .server import RootstockServer


class RootstockCalculator(Calculator):
    """
    ASE calculator that runs MLIPs in an isolated subprocess.

    This calculator:
    1. Spawns a worker process that loads the MLIP
    2. Communicates via i-PI protocol over Unix sockets
    3. Keeps the worker alive across calculations (no startup overhead)

    Example:
        from ase.build import bulk
        from rootstock import RootstockCalculator

        atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

        with RootstockCalculator(
            environment="mace_env",
            model="mace-mp-0",
            device="cuda",
            root="/shared/rootstock",
        ) as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())
            print(atoms.get_forces())
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        environment: str,
        model: str,
        device: str = "cuda",
        root: str | Path | None = None,
        log=None,
        **kwargs,
    ):
        """
        Initialize the Rootstock calculator.

        Args:
            environment: Name of registered environment or path to environment file.
            model: Model identifier (e.g., "mace-mp-0", "medium", or path to weights)
            device: Device for MLIP ("cuda", "cuda:0", "cpu")
            root: Root directory for environments and cache. Required for named
                  environments, optional for path-based environments.
            log: Optional file object for logging
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)

        self.environment = environment
        self.model = model
        self.device = device
        self.root = Path(root) if root else None
        self.log = log

        # Generate unique socket name to avoid conflicts
        self._socket_name = f"rootstock_{uuid.uuid4().hex[:8]}"
        self._server: RootstockServer | None = None

    def _ensure_server(self):
        """Start server if not already running."""
        if self._server is None:
            from .environment import EnvironmentManager

            env_manager = EnvironmentManager(root=self.root)
            env_path = env_manager.resolve_environment(self.environment)

            self._server = RootstockServer(
                environment_path=env_path,
                model=self.model,
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
