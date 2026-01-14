#!/usr/bin/env python
"""
Rootstock worker process.

This runs in an isolated subprocess and:
1. Loads an MLIP (e.g., MACE)
2. Connects to the server via Unix socket
3. Receives positions, calculates forces, sends results back
4. Persists across multiple calculations (no startup overhead per calculation)

Usage (legacy):
    python worker.py --socket rootstock --model medium

Usage (v0.2 via wrapper):
    The worker is spawned via a generated wrapper script that calls run_worker().
"""

import argparse
import json
import sys
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

# Handle imports for both direct invocation and package usage
try:
    from .protocol import (
        IPIProtocol,
        SocketClosed,
        connect_unix_socket,
        create_unix_socket_path,
    )
except ImportError:
    from protocol import (
        IPIProtocol,
        SocketClosed,
        connect_unix_socket,
        create_unix_socket_path,
    )

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class MLIPWorker:
    """
    Worker that runs an MLIP calculator and communicates via i-PI protocol.
    
    The worker acts as an i-PI client:
    1. Connect to server
    2. Report READY status
    3. Receive positions via POSDATA
    4. Calculate energy/forces
    5. Report HAVEDATA status
    6. Send results via FORCEREADY
    7. Loop back to step 2
    """
    
    def __init__(
        self,
        socket_name: str,
        model: str = "medium",
        device: str = "cuda",
        log=None,
        calculator: Optional["Calculator"] = None,
    ):
        """
        Initialize the worker.

        Args:
            socket_name: Name of Unix socket to connect to
            model: MACE model name or path
            device: Device to run on ("cuda" or "cpu")
            log: Optional file object for logging
            calculator: Optional pre-loaded calculator (skips _load_calculator if provided)
        """
        self.socket_name = socket_name
        self.socket_path = create_unix_socket_path(socket_name)
        self.model = model
        self.device = device
        self.log = log

        self._calculator = calculator  # May be pre-loaded
        self._socket = None
        self._protocol = None
        self._atoms = None  # Cache ASE Atoms object

        # Atomic species info from INIT message (v0.2)
        self._atomic_numbers: Optional[list[int]] = None
        self._pbc: Optional[list[bool]] = None
    
    def _log(self, msg):
        if self.log:
            print(f"[Worker] {msg}", file=self.log, flush=True)
    
    def _load_calculator(self):
        """Load the MACE calculator (legacy mode only)."""
        if self._calculator is not None:
            self._log("Using pre-loaded calculator")
            return

        self._log(f"Loading MACE model: {self.model}")

        from mace.calculators import mace_mp

        # mace_mp returns a pre-configured calculator for the foundation models
        self._calculator = mace_mp(
            model=self.model,
            device=self.device,
            default_dtype="float32",
        )

        self._log("MACE calculator loaded")
    
    def _connect(self):
        """Connect to the server."""
        self._log(f"Connecting to {self.socket_path}")
        self._socket = connect_unix_socket(self.socket_path)
        self._protocol = IPIProtocol(self._socket, log=self.log)
        self._log("Connected")
    
    def _create_atoms(self, positions: np.ndarray, cell: np.ndarray):
        """
        Create or update ASE Atoms object.

        On first call, creates a new Atoms object.
        On subsequent calls, updates positions and cell in place.

        Uses atomic numbers from INIT message if available (v0.2),
        otherwise falls back to Cu for backward compatibility.
        """
        from ase import Atoms

        if self._atoms is None or len(self._atoms) != len(positions):
            # Need to create new Atoms object
            # Use atomic numbers from INIT if available, else default to Cu
            if self._atomic_numbers is not None:
                numbers = self._atomic_numbers
            else:
                numbers = [29] * len(positions)  # Cu fallback for legacy mode

            pbc = self._pbc if self._pbc is not None else [True, True, True]

            self._atoms = Atoms(
                numbers=numbers,
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
            self._atoms.calc = self._calculator
        else:
            # Update existing object (faster - reuses neighbor lists etc.)
            self._atoms.positions = positions
            self._atoms.cell = cell

        return self._atoms
    
    def _calculate(
        self, positions: np.ndarray, cell: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Run MLIP calculation.
        
        Returns:
            energy: Potential energy in eV
            forces: Nx3 forces in eV/Angstrom
            virial: 3x3 virial tensor in eV
        """
        atoms = self._create_atoms(positions, cell)
        
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        # Calculate virial from stress
        # stress is in eV/Å³, virial = -stress * volume
        try:
            stress = atoms.get_stress(voigt=False)  # 3x3 tensor
            volume = atoms.get_volume()
            virial = -stress * volume
        except Exception:
            # Some calculators don't support stress
            virial = np.zeros((3, 3))
        
        return energy, forces, virial
    
    def run(self):
        """
        Main loop - receive positions, calculate, send results.
        
        This implements the i-PI client state machine:
        - NEEDINIT -> receive INIT -> READY
        - READY -> receive POSDATA -> calculate -> HAVEDATA
        - HAVEDATA -> receive GETFORCE -> send FORCEREADY -> NEEDINIT
        """
        self._load_calculator()
        self._connect()
        
        state = "NEEDINIT"
        energy = None
        forces = None
        virial = None
        
        self._log("Entering main loop")
        
        try:
            while True:
                # Wait for message from server
                try:
                    msg = self._protocol.recvmsg()
                except SocketClosed:
                    self._log("Server closed connection")
                    break
                
                if msg == "EXIT":
                    self._log("Received EXIT, shutting down")
                    break
                
                elif msg == "STATUS":
                    # Report current state
                    if state == "NEEDINIT":
                        self._protocol.sendmsg("NEEDINIT")
                    elif state == "READY":
                        self._protocol.sendmsg("READY")
                    elif state == "HAVEDATA":
                        self._protocol.sendmsg("HAVEDATA")
                
                elif msg == "INIT":
                    # Receive initialization with atomic species info (v0.2)
                    bead_index, init_bytes = self._protocol.recv_init()

                    # Parse JSON from init_bytes if present
                    if init_bytes and init_bytes != b"\x00":
                        try:
                            init_data = json.loads(init_bytes.decode("utf-8"))
                            self._atomic_numbers = init_data.get("numbers")
                            self._pbc = init_data.get("pbc", [True, True, True])
                            self._log(
                                f"Received INIT (bead={bead_index}, "
                                f"atoms={len(self._atomic_numbers) if self._atomic_numbers else 0})"
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            self._log(f"Warning: Failed to parse INIT data: {e}")
                    else:
                        self._log(f"Received INIT (bead={bead_index}, no species data)")

                    state = "READY"
                
                elif msg == "POSDATA":
                    # Receive atomic positions
                    if state not in ("READY", "NEEDINIT"):
                        self._log(f"Warning: POSDATA in state {state}")
                    
                    cell, positions = self._protocol.recv_posdata()
                    self._log(f"Received POSDATA: {len(positions)} atoms")
                    
                    # Calculate energy and forces
                    energy, forces, virial = self._calculate(positions, cell)
                    self._log(f"Calculated: E={energy:.6f} eV")
                    
                    state = "HAVEDATA"
                
                elif msg == "GETFORCE":
                    # Send results
                    if state != "HAVEDATA":
                        raise RuntimeError(f"GETFORCE in state {state}")
                    
                    self._protocol.send_forceready(energy, forces, virial)
                    self._log("Sent FORCEREADY")
                    
                    state = "NEEDINIT"
                
                else:
                    self._log(f"Unknown message: {msg}")
        
        finally:
            if self._socket:
                self._socket.close()
            self._log("Worker shutdown complete")
    

def run_worker(
    setup_fn: Callable[[str, str], "Calculator"],
    model: str,
    device: str,
    socket_path: str,
    log=None,
):
    """
    Run worker with a provided setup function.

    This is the entry point used by generated wrapper scripts (v0.2).
    The setup function is called once to create the calculator, which
    is then reused for all subsequent calculations.

    Args:
        setup_fn: Function that takes (model, device) and returns an ASE calculator
        model: Model identifier to pass to setup_fn
        device: Device string to pass to setup_fn
        socket_path: Full Unix socket path to connect to
        log: Optional logging file object
    """
    if log:
        print(f"[Worker] Calling setup({model!r}, {device!r})", file=log, flush=True)

    # Load calculator via the setup function
    calculator = setup_fn(model, device)

    if log:
        print(f"[Worker] Calculator loaded: {type(calculator).__name__}", file=log, flush=True)

    # Extract socket name from path (e.g., /tmp/ipi_rootstock_abc -> rootstock_abc)
    socket_name = socket_path.replace("/tmp/ipi_", "")

    worker = MLIPWorker(
        socket_name=socket_name,
        model=model,
        device=device,
        log=log,
        calculator=calculator,
    )
    worker.run()


def main():
    """Legacy entry point for direct worker invocation."""
    parser = argparse.ArgumentParser(description="Rootstock MLIP worker")
    parser.add_argument("--socket", required=True, help="Unix socket name")
    parser.add_argument("--model", default="medium", help="MACE model name")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log = sys.stderr if args.verbose else None

    worker = MLIPWorker(
        socket_name=args.socket,
        model=args.model,
        device=args.device,
        log=log,
    )
    worker.run()


if __name__ == "__main__":
    main()
