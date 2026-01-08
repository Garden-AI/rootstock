"""
ASE-compatible calculator that delegates to an MLIP worker process.

This is the main user-facing interface for Rootstock.
"""

import uuid
from typing import Optional

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
        
        with RootstockCalculator(model="medium") as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())
            print(atoms.get_forces())
    """
    
    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    
    def __init__(
        self,
        model: str = "medium",
        device: str = "cuda",
        worker_python: Optional[str] = None,
        log=None,
        **kwargs,
    ):
        """
        Initialize the Rootstock calculator.
        
        Args:
            model: MACE model name (e.g., "small", "medium", "large")
            device: Device for MLIP ("cuda" or "cpu")
            worker_python: Python executable for worker process. If None, uses
                          the same Python as the current process.
            log: Optional file object for logging
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)
        
        self.model = model
        self.device = device
        self.worker_python = worker_python
        self.log = log
        
        # Generate unique socket name to avoid conflicts
        self._socket_name = f"rootstock_{uuid.uuid4().hex[:8]}"
        self._server: Optional[RootstockServer] = None
    
    def _ensure_server(self):
        """Start server if not already running."""
        if self._server is None:
            self._server = RootstockServer(
                socket_name=self._socket_name,
                worker_python=self.worker_python,
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
