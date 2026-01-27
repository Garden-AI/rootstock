#!/usr/bin/env python
"""
Rootstock worker process.

This runs in an isolated subprocess and:
1. Loads an MLIP (e.g., MACE)
2. Connects to the server via Unix socket
3. Receives positions, calculates forces, sends results back
4. Persists across multiple calculations (no startup overhead per calculation)

The worker is spawned via a generated wrapper script that calls run_worker().
"""

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .protocol import (
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
        calculator: "Calculator",
        log=None,
    ):
        """
        Initialize the worker.

        Args:
            socket_name: Name of Unix socket to connect to
            calculator: Pre-loaded ASE calculator
            log: Optional file object for logging
        """
        self.socket_name = socket_name
        self.socket_path = create_unix_socket_path(socket_name)
        self.log = log

        self._calculator = calculator
        self._socket = None
        self._protocol = None
        self._atoms = None  # Cache ASE Atoms object

        # Atomic species info from INIT message
        self._atomic_numbers: list[int] | None = None
        self._pbc: list[bool] | None = None

    def _log(self, msg):
        if self.log:
            print(f"[Worker] {msg}", file=self.log, flush=True)

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
        """
        from ase import Atoms

        if self._atoms is None or len(self._atoms) != len(positions):
            # Need to create new Atoms object
            if self._atomic_numbers is None:
                raise RuntimeError(
                    "No atomic numbers received. Server must send INIT with species data."
                )

            self._atoms = Atoms(
                numbers=self._atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=self._pbc if self._pbc is not None else [True, True, True],
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
                    # Receive initialization with atomic species info
                    bead_index, init_bytes = self._protocol.recv_init()

                    # Parse JSON from init_bytes
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

    This is the entry point used by generated wrapper scripts.
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
        calculator=calculator,
        log=log,
    )
    worker.run()
