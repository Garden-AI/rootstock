"""MLIPx integration.

Expose a Rootstock-hosted checkpoint as a model that
`MLIPx <https://github.com/basf/mlipx>`_ can evaluate. MLIPx consumes any
object satisfying its ``NodeWithCalculator`` protocol (a ``get_calculator()``
returning an ASE calculator, plus an optional ``get_spec()``). Because
``RootstockCalculator`` is already an ASE calculator, no MLIPx-side code is
needed; this node just wraps it with zntrack-tracked parameters so runs are
reproducible and MLIPx's comparison tables get real metadata.

Requires the optional ``mlipx`` extra::

    pip install "rootstock[mlipx]"

Usage inside an MLIPx recipe's ``models.py``::

    from rootstock.integrations.mlipx import RootstockMLIPxModel

    MODELS = {
        "mace": RootstockMLIPxModel(
            checkpoint="mace-mp-0-medium", cluster="sophia", device="cuda"
        ),
        "uma": RootstockMLIPxModel(checkpoint="uma-s-1p1", cluster="sophia", device="cuda"),
    }

Locally (no cluster), point at an install root instead::

    RootstockMLIPxModel(checkpoint="mace-mp-0-medium", root="/path/to/rootstock", device="cpu")

The target environment must already be built with ``rootstock install``.
"""

from __future__ import annotations

import typing as t

try:
    import zntrack
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "RootstockMLIPxModel requires the optional 'mlipx' extra. "
        'Install it with: pip install "rootstock[mlipx]"'
    ) from exc

from ase.calculators.calculator import Calculator

from ..calculator import RootstockCalculator


class RootstockMLIPxModel(zntrack.Node):
    """A Rootstock-hosted checkpoint, usable anywhere MLIPx expects a model.

    Satisfies MLIPx's ``NodeWithCalculator`` protocol. Unlike a bare
    ``mlipx.GenericASECalculator`` config entry, the parameters below are
    tracked by zntrack (reproducible, diffable) and ``get_spec`` reports real
    per-model metadata.
    """

    checkpoint: str = zntrack.params()
    cluster: str | None = zntrack.params(None)
    root: str | None = zntrack.params(None)
    device: str = zntrack.params("cpu")  # RootstockCalculator defaults to cuda; we don't.
    setup_kwargs: dict | None = zntrack.params(None)

    def run(self) -> None:
        # Pure calculator provider; `run` exists to satisfy zntrack and to
        # validate the cluster/root choice early.
        if (self.cluster is None) == (self.root is None):
            raise ValueError("Specify exactly one of `cluster` or `root`.")

    def get_calculator(self, **kwargs) -> Calculator:
        """Return a ``RootstockCalculator`` for this checkpoint.

        The worker subprocess spawns lazily on the first ``calculate()`` and
        persists across frames (Rootstock's design). MLIPx does not call
        ``close()``, so teardown falls to the calculator's ``__del__``. That is
        fine for a single evaluation node; for a large comparison graph holding
        many models at once, close them explicitly to avoid leaked workers.
        """
        return RootstockCalculator(
            checkpoint=self.checkpoint,
            cluster=self.cluster,
            root=self.root,
            device=self.device,
            setup_kwargs=self.setup_kwargs or {},
            **kwargs,
        )

    def get_spec(self) -> dict | None:
        """Best-effort model metadata, shaped like ``mlipx.spec.MLIPSpec``.

        Enriches with the resolved environment name when an install root is
        reachable; otherwise falls back to the node's parameters. Training-data
        provenance is left empty rather than guessed.
        """
        metadata: dict[str, t.Any] = {
            "name": self.checkpoint,
            "checkpoint": self.checkpoint,
            "backend": "rootstock",
            "location": self.cluster or self.root,
            "device": self.device,
        }
        try:
            from ..clusters import get_root_for_cluster
            from ..environment import resolve_checkpoint

            root = get_root_for_cluster(self.cluster) if self.cluster else self.root
            if root is not None:
                metadata["env"] = resolve_checkpoint(root, self.checkpoint).env_name
        except Exception:
            pass

        return {"metadata": metadata, "data": None}
