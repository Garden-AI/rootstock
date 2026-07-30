"""atomate2 integration.

Expose a Rootstock-hosted checkpoint as a force-field Maker that
`atomate2 <https://github.com/materialsproject/atomate2>`_ workflows can drive.
atomate2 already accepts arbitrary ASE calculators: ``ForceFieldMixin`` will
dynamically load any import path handed to ``calculator_meta``. So no
atomate2-side change is needed. What this module adds is ergonomics, worker
lifecycle handling, and provenance. atomate2 caches the calculator on the Maker
and never calls ``close()``, so the i-PI worker would otherwise outlive the job.

Requires the optional ``atomate2`` extra (Python 3.11+, since atomate2 does not
support 3.10)::

    pip install "rootstock[atomate2]"

Usage::

    from jobflow import run_locally

    from rootstock.integrations.atomate2 import RootstockAtomate2RelaxMaker

    maker = RootstockAtomate2RelaxMaker(
        checkpoint="mace-mp-0-medium",
        cluster="sophia",
        device="cuda",
    )
    run_locally(maker.make(structure), create_folders=True)

Locally (no cluster), point at an install root instead::

    RootstockAtomate2RelaxMaker(
        checkpoint="mace-mp-0-medium",
        root="/path/to/rootstock",
        device="cpu",
    )

These Makers drop into any atomate2 flow that takes a ``ForceFieldRelaxMaker``
or ``ForceFieldStaticMaker`` (phonons, elastic, EOS, QHA, and so on)::

    from atomate2.forcefields.flows.phonons import PhononMaker

    relax = RootstockAtomate2RelaxMaker(checkpoint="mace-mp-0-medium", cluster="sophia")
    static = RootstockAtomate2StaticMaker(checkpoint="mace-mp-0-medium", cluster="sophia")

    PhononMaker(
        bulk_relax_maker=relax,
        static_energy_maker=static,
        phonon_displacement_maker=static,
    )

On ``force_field_name``: atomate2's ``MLFF`` enum names *models*, not execution
backends, and rejects unknown strings. A Rootstock-hosted MACE is still MACE, so
these Makers leave ``force_field_name`` at the ``MLFF.Forcefield`` placeholder
(which also stops atomate2 from padding ``calculator_kwargs`` with some other
MLFF's defaults) and identify the backend through ``calculator_meta`` instead.

The target environment must already be built with ``rootstock install``.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

try:
    from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "The Rootstock atomate2 integration requires the optional 'atomate2' extra "
        "and Python 3.11 or newer (atomate2 does not support 3.10). "
        'Install it with: pip install "rootstock[atomate2]"'
    ) from exc

from ..calculator import RootstockCalculator

if t.TYPE_CHECKING:
    from pathlib import Path

    from ase.calculators.calculator import Calculator
    from atomate2.ase.schemas import AseResult
    from pymatgen.core import Molecule, Structure

# Points atomate2's schema naming and provenance at RootstockCalculator.
# `_get_calculator` is overridden below, so this string never constructs
# anything; it only sets `ase_calculator_name` on the task document.
_CALCULATOR_META = "rootstock.calculator.RootstockCalculator"


@dataclass
class _RootstockAtomate2Mixin:
    """Shared Rootstock wiring for the atomate2 force-field Makers."""

    checkpoint: str = ""
    cluster: str | None = None
    root: str | Path | None = None
    cache_root: str | Path | None = None
    device: str = "cpu"  # RootstockCalculator defaults to cuda; we don't.
    setup_kwargs: dict = field(default_factory=dict)
    close_worker: bool = True

    def __post_init__(self) -> None:
        """Validate Rootstock args on top of atomate2's own validation."""
        super().__post_init__()

        if not self.checkpoint:
            raise ValueError("`checkpoint` is required (e.g. 'mace-mp-0-medium').")
        if (self.cluster is None) == (self.root is None):
            raise ValueError("Specify exactly one of `cluster` or `root`.")

    def _get_calculator(self) -> Calculator:
        """Build a RootstockCalculator instead of importing an MLIP locally.

        atomate2 caches the return value on `self._calculator` behind the
        `calculator` property, so one worker serves every ionic step of a job.
        """
        return RootstockCalculator(
            checkpoint=self.checkpoint,
            cluster=self.cluster,
            root=self.root,
            cache_root=self.cache_root,
            device=self.device,
            setup_kwargs=self.setup_kwargs,
            **self.calculator_kwargs,
        )

    def close(self) -> None:
        """Stop the worker and drop atomate2's cached calculator."""
        calc = getattr(self, "_calculator", None)
        if calc is not None:
            calc.close()
            self._calculator = None

    def run_ase(
        self,
        mol_or_struct: Structure | Molecule,
        prev_dir: str | Path | None = None,
    ) -> AseResult:
        """Run the atomate2 job, then stop the worker.

        atomate2 never calls `close()` on a calculator, so without this the i-PI
        worker and its device allocation survive for the life of the process.

        `make` calls `run_ase` once per structure, so in batch mode this restarts
        the worker for each structure. Pass `close_worker=False` to keep the
        worker hot across a batch and call `close()` yourself when done.
        """
        try:
            return super().run_ase(mol_or_struct, prev_dir=prev_dir)
        finally:
            if self.close_worker:
                self.close()


@dataclass
class RootstockAtomate2RelaxMaker(_RootstockAtomate2Mixin, ForceFieldRelaxMaker):
    """Relax a structure or molecule with a Rootstock-hosted MLIP."""

    name: str = "Rootstock relax"
    calculator_meta: str = _CALCULATOR_META


@dataclass
class RootstockAtomate2StaticMaker(_RootstockAtomate2Mixin, ForceFieldStaticMaker):
    """Single-point energy, forces, and stress from a Rootstock-hosted MLIP."""

    name: str = "Rootstock static"
    calculator_meta: str = _CALCULATOR_META
