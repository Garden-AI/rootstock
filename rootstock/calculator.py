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
from .server import RootstockServer


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
            cluster="della",
            model="mace",
            checkpoint="medium",
            device="cuda",
        ) as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())

        # Forward extra kwargs to the env's setup() function:
        with RootstockCalculator(
            cluster="della",
            model="uma",
            checkpoint="uma-s-1p1",
            device="cuda",
            setup_kwargs={"task": "omol"},
        ) as calc:
            atoms.calc = calc
            print(atoms.get_potential_energy())

    Note:
        Environments must be pre-built using `rootstock build` before use.
        Run: rootstock build mace_env --root /path/to/rootstock
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str,
        checkpoint: str,
        cluster: str | None = None,
        root: str | Path | None = None,
        cache_root: str | Path | None = None,
        device: str = "cuda",
        setup_kwargs: dict | None = None,
        log=None,
        **kwargs,
    ):
        """
        Initialize the Rootstock calculator.

        Args:
            model: Environment family name (e.g. "mace", "uma", "tensornet").
                   Maps to {model}_env environment.
            checkpoint: Specific checkpoint/weights to load. Passed to the
                        environment's setup() as the model argument. Required.
            cluster: Known cluster name (e.g., "della", "perlmutter"). Mutually
                     exclusive with root. The cluster's registered cache_root is
                     used unless `cache_root` is also passed.
            root: Path to rootstock install directory. Mutually exclusive with cluster.
            cache_root: Optional separate root for the model-weight cache and
                        redirected HOME. Defaults to the cluster's registered
                        cache_root, or to ``root`` if no cluster is in play.
            device: PyTorch device ("cuda", "cuda:0", "cpu")
            setup_kwargs: Extra keyword arguments forwarded to the env's setup()
                          function. May not contain "model" or "device" — those
                          are passed at the top level.
            log: Optional file object for logging
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)

        if setup_kwargs is None:
            setup_kwargs = {}
        reserved = {"model", "device"} & setup_kwargs.keys()
        if reserved:
            raise TypeError(
                f"setup_kwargs cannot contain reserved keys {sorted(reserved)}; "
                "pass them at the top level instead."
            )

        self.device = device
        self.setup_kwargs = setup_kwargs
        self.log = log

        # Resolve install root and cache root.
        # Resolution: explicit kwarg > cluster registry default > install root.
        if cluster is not None and root is not None:
            raise ValueError("Cannot specify both 'cluster' and 'root'")

        if cluster is not None:
            cluster_info = get_cluster(cluster)
            self.root = cluster_info.root
            self.cache_root = (
                Path(cache_root) if cache_root is not None else cluster_info.resolved_cache_root
            )
        elif root is not None:
            self.root = Path(root)
            self.cache_root = Path(cache_root) if cache_root is not None else self.root
        else:
            raise ValueError("Must specify either 'cluster' or 'root'")

        self.env_name = f"{model}_env"
        self.model_arg = checkpoint

        # Verify environment is built
        env_python = self.root / "envs" / self.env_name / "bin" / "python"
        if not env_python.exists():
            envs_dir = self.root / "envs"
            if envs_dir.exists():
                available = [p.name for p in envs_dir.iterdir() if p.is_dir()]
            else:
                available = []
            raise RuntimeError(
                f"Environment '{self.env_name}' not built at {self.root}/envs/{self.env_name}/\n"
                f"Run: rootstock build {self.env_name} --root {self.root}\n"
                f"Available environments: {available}"
            )

        # Generate unique socket name to avoid conflicts
        self._socket_name = f"rootstock_{uuid.uuid4().hex[:8]}"
        self._server: RootstockServer | None = None

    def _ensure_server(self):
        """Start server if not already running."""
        if self._server is None:
            self._server = RootstockServer(
                env_name=self.env_name,
                model=self.model_arg,
                device=self.device,
                socket_name=self._socket_name,
                root=self.root,
                cache_root=self.cache_root,
                log=self.log,
                setup_kwargs=self.setup_kwargs,
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
