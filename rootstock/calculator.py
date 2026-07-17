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

from .clusters import get_cluster
from .config import resolve_default_root
from .environment import find_env_for_checkpoint
from .layout import ensure_layout_compatible, resolve_cache_root
from .server import RootstockServer, WorkerDiedError


class RootstockCalculator(Calculator):
    """
    ASE calculator that runs MLIPs in a pre-built isolated environment.

    This calculator:
    1. Spawns a worker process using a pre-built virtual environment
    2. Communicates via i-PI protocol over Unix sockets
    3. Keeps the worker alive across calculations (no startup overhead)

    Example:
        from ase.build import bulk
        from rootstock import RootstockCalculator

        atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)

        with RootstockCalculator(
            checkpoint="mace-mp-0-medium",
            cluster="della",
            device="cuda",
        ) as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())

        # Forward extra kwargs to the env's setup() function:
        with RootstockCalculator(
            checkpoint="uma-s-1p1",
            cluster="della",
            device="cuda",
            setup_kwargs={"task": "omol"},
        ) as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())

    Note:
        Environments must be pre-built with `rootstock install` before use.
        The hosting env is resolved automatically by walking the installed envs
        and matching the checkpoint id against each env file's `CHECKPOINTS`.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        checkpoint: str,
        cluster: str | None = None,
        root: str | Path | None = None,
        cache_root: str | Path | None = None,
        device: str = "cuda",
        setup_kwargs: dict | None = None,
        timeout: float = 600.0,
        **kwargs,
    ):
        """
        Initialize the Rootstock calculator.

        Diagnostics go to stdlib logging under the ``rootstock`` namespace:
        server lifecycle on ``rootstock.server`` (INFO/DEBUG), the wire trace
        on ``rootstock.protocol`` (DEBUG). Enable with e.g.
        ``logging.basicConfig(level=logging.DEBUG)``.

        Args:
            checkpoint: Canonical checkpoint id (e.g., "mace-mp-0-medium",
                        "uma-s-1p1"). Required. The hosting env is resolved
                        from the installed envs at ``root``.
            cluster: Known cluster name (e.g., "della", "perlmutter"). Mutually
                     exclusive with root; a name -> install-path bootstrap.
            root: Path to rootstock install directory. Mutually exclusive with
                  cluster. When neither is given, the ROOTSTOCK_ROOT
                  environment variable and then the ``root`` in
                  ~/.config/rootstock/config.toml are used — the same
                  fallback the CLI applies.
            cache_root: Optional override for the model-weight cache and
                        redirected HOME. When omitted, the install's own
                        declaration ({root}/layout.json) decides, falling back
                        to the cluster registry for legacy roots, then to
                        ``root`` — the same resolution the CLI uses.
            device: PyTorch device ("cuda", "cuda:0", "cpu")
            setup_kwargs: Extra keyword arguments forwarded to the env's setup()
                          function. May not contain "checkpoint" or "device" —
                          those are passed at the top level.
            timeout: Socket timeout in seconds for worker operations. The
                     default (600 s) matches checkpoint verification, so the
                     first real force call — which may pay for torch.compile
                     or large neighbor lists — runs under the same envelope
                     verification exercised.
            **kwargs: Additional arguments passed to ASE Calculator
        """
        # ASE's Calculator quietly absorbs unknown kwargs as parameters, so a
        # 0.x caller passing the removed log= would silently lose their logs.
        if "log" in kwargs:
            raise TypeError(
                "RootstockCalculator no longer takes 'log'. Client-side "
                "diagnostics use stdlib logging — e.g. "
                "logging.basicConfig(level=logging.DEBUG) — and worker "
                "verbosity is controlled by the ROOTSTOCK_WORKER_LOG env var."
            )
        super().__init__(**kwargs)

        if setup_kwargs is None:
            setup_kwargs = {}
        reserved = {"checkpoint", "device"} & setup_kwargs.keys()
        if reserved:
            raise TypeError(
                f"setup_kwargs cannot contain reserved keys {sorted(reserved)}; "
                "pass them at the top level instead."
            )

        self.checkpoint = checkpoint
        self.device = device
        self.setup_kwargs = setup_kwargs
        self.timeout = timeout

        # Resolve the install root: the cluster name is only a name -> path
        # bootstrap. Everything else about the install (including where its
        # cache lives) is read from the install itself, identically for both
        # entry points — see resolve_cache_root.
        if cluster is not None and root is not None:
            raise ValueError("Cannot specify both 'cluster' and 'root'")

        if cluster is not None:
            self.root = get_cluster(cluster).root
        elif root is not None:
            self.root = Path(root)
        else:
            default_root = resolve_default_root()
            if default_root is None:
                raise ValueError(
                    "Must specify 'cluster' or 'root' (or set the "
                    "ROOTSTOCK_ROOT environment variable, or configure root "
                    "in ~/.config/rootstock/config.toml)"
                )
            self.root = default_root

        self.cache_root = resolve_cache_root(self.root, explicit=cache_root)

        # A root laid out by a newer rootstock may not be readable by this
        # client's conventions — fail with "upgrade rootstock", not a
        # misleading resolution error.
        ensure_layout_compatible(self.root)

        # Resolve env name from the canonical checkpoint id by walking the
        # installed envs at self.root. Raises CheckpointNotFoundError with a
        # helpful listing if no env declares this id.
        self.env_name, _ = find_env_for_checkpoint(self.root, checkpoint)

        # Generate unique socket name to avoid conflicts
        self._socket_name = f"rootstock_{uuid.uuid4().hex[:8]}"
        self._server: RootstockServer | None = None

    def _ensure_server(self):
        """Start server if not already running."""
        if self._server is None:
            self._server = RootstockServer(
                env_name=self.env_name,
                checkpoint=self.checkpoint,
                device=self.device,
                socket_name=self._socket_name,
                root=self.root,
                cache_root=self.cache_root,
                setup_kwargs=self.setup_kwargs,
                timeout=self.timeout,
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
        try:
            energy, forces, virial = self._server.calculate(
                positions=self.atoms.positions,
                cell=np.array(self.atoms.cell),
                atomic_numbers=self.atoms.numbers,
                pbc=list(self.atoms.pbc),
            )
        except WorkerDiedError:
            # A dead worker can't serve the next call either. Tear the server
            # down so a subsequent calculation starts a fresh one, instead of
            # this instance being permanently bricked by one GPU OOM mid-MD.
            # No automatic retry — the same configuration would likely fail
            # the same way; the caller decides whether to try again.
            self.close()
            raise

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
