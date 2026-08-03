"""
Checkpoint verification for Rootstock.

A single helper that spawns a worker against a (env, checkpoint, device,
setup_kwargs) combination, runs one forward pass on a hardcoded test structure,
and asserts the result is finite and non-degenerate.

Used by both ``rootstock add`` (one checkpoint) and ``rootstock smoke-test``
(all of them). The smoke test only asks "did the computational pipeline
produce a finite, non-degenerate result?" — not "is the answer chemically
meaningful?". A water molecule in vacuum is out-of-domain for inorganic-only
MLIPs, but those models still return finite numbers; that's all we need.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


# Threshold below which we consider forces "all zero" — guards against the
# silent failure where a model returns zeros for everything.
_FORCE_ZERO_THRESHOLD = 1e-8

# Tolerances for results_mismatch: loose enough for GPU nondeterminism and
# minor loader differences between setup() and setup_from_path() (e.g. dtype
# defaults), tight enough that a different model cannot pass — wrong weights
# move energies by orders of magnitude more than either.
_ENERGY_ATOL = 1e-4  # eV
_FORCES_ATOL = 1e-3  # eV/Å


def _smoke_test_atoms() -> "Atoms":  # noqa: UP037 — Atoms is TYPE_CHECKING-only
    """Hardcoded H2O-in-a-box used as input for every smoke test."""
    from ase import Atoms

    atoms = Atoms(
        "H2O",
        positions=[
            [0.00, 0.00, 0.00],
            [0.96, 0.00, 0.00],
            [0.24, 0.93, 0.00],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    # OMol-trained checkpoints (eSEN, Orb v3) require these. Harmless for
    # everyone else.
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    # Break any accidental symmetry so forces aren't trivially zero.
    atoms.positions[1, 1] += 0.05
    return atoms


def verify_checkpoint(
    root: Path,
    env_name: str,
    checkpoint: str,
    device: str,
    setup_kwargs: dict | None = None,
    cache_root: Path | None = None,
    checkpoint_path: str | None = None,
    weights_capture_path: str | None = None,
    results: dict | None = None,
) -> tuple[bool, str | None]:
    """
    Run a single forward pass to verify a checkpoint loads and computes.

    Args:
        root: Rootstock install root.
        env_name: Name of pre-built environment (e.g., "mace").
        checkpoint: Canonical checkpoint id passed to setup().
        device: PyTorch device (e.g., "cuda", "cpu").
        setup_kwargs: Extra keyword arguments forwarded to setup().
        cache_root: Optional separate root for the model-weight cache and
                    redirected HOME. Defaults to ``root``.
        checkpoint_path: Path to a local (user-registered) weights file; the
                    worker then loads via setup_from_path() instead of setup().
        weights_capture_path: When set, the worker writes the weight files
                    its setup() touched to this path as JSON (#177); the
                    caller reads it afterwards — a failed load writes nothing.
        results: When a dict is passed and verification succeeds, it receives
                    the computed ``energy``/``forces``/``virial`` so the
                    caller can compare runs (see ``results_mismatch``).

    Returns:
        (success, error_message). On success, error_message is None.
        On failure, success is False and error_message is a short string
        describing what went wrong.
    """
    from .server import RootstockServer

    setup_kwargs = setup_kwargs or {}
    atoms = _smoke_test_atoms()
    socket_name = f"rootstock_verify_{uuid.uuid4().hex[:8]}"

    server = RootstockServer(
        env_name=env_name,
        checkpoint=checkpoint,
        device=device,
        socket_name=socket_name,
        root=Path(root),
        cache_root=Path(cache_root) if cache_root is not None else None,
        setup_kwargs=setup_kwargs,
        timeout=600.0,
        checkpoint_path=checkpoint_path,
        weights_capture_path=weights_capture_path,
        # Verification sessions are synthetic — the nightly smoke-test cron
        # would otherwise spool one fake "session" per checkpoint per night,
        # and rollups merge irreversibly.
        usage_client=None,
    )

    try:
        server.start()
        energy, forces, virial = server.calculate(
            positions=atoms.positions,
            cell=np.array(atoms.cell),
            atomic_numbers=atoms.numbers,
            pbc=list(atoms.pbc),
            info=dict(atoms.info),
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            server.stop()
        except Exception:
            pass

    n_atoms = len(atoms)
    if not np.isfinite(energy):
        return False, f"non-finite energy: {energy!r}"
    if forces.shape != (n_atoms, 3):
        return False, f"forces shape {forces.shape} != ({n_atoms}, 3)"
    if not np.all(np.isfinite(forces)):
        return False, "non-finite values in forces"
    if np.linalg.norm(forces) <= _FORCE_ZERO_THRESHOLD:
        return False, "forces are all (near-)zero — model likely returned zeros"
    if not np.all(np.isfinite(virial)):
        return False, "non-finite values in virial"

    if results is not None:
        results["energy"] = float(energy)
        results["forces"] = np.asarray(forces, dtype=float)
        results["virial"] = np.asarray(virial, dtype=float)
    return True, None


def results_mismatch(baseline: dict, candidate: dict) -> str | None:
    """
    Compare two successful ``verify_checkpoint`` results (``results`` dicts)
    from the same structure on the same device.

    Used by smoke-test's custom-weights leg (#200): the same checkpoint loaded
    via setup() and via setup_from_path(weights file) must agree — anything
    beyond tolerance means the weights= path loaded something else.

    Returns a short description of the first difference, or None when they
    agree within tolerance.
    """
    d_energy = abs(candidate["energy"] - baseline["energy"])
    if d_energy > _ENERGY_ATOL:
        return f"energy differs by {d_energy:.3e} eV (tolerance {_ENERGY_ATOL:.0e})"
    d_forces = float(np.max(np.abs(candidate["forces"] - baseline["forces"])))
    if d_forces > _FORCES_ATOL:
        return f"max force component differs by {d_forces:.3e} eV/Å (tolerance {_FORCES_ATOL:.0e})"
    return None
