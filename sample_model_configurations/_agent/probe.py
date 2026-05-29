"""
No-IPC probe: imports a config file's `setup()`, instantiates a calculator, runs
one forward pass, and prints structured stage markers so an outer agent (or
human) can attribute time and diagnose hangs.

Designed to run inside a Modal image whose deps were installed from the
config's PEP 723 metadata. The config file is mounted into the image at a
known path; this script imports `setup` from that path via importlib.

    python probe.py --config /workshop/config.py \\
        --checkpoint <name> --device cuda --system molecule

Output contract: every meaningful stage prints `STAGE: <name> elapsed=<sec>`.
If wall time exceeds the last stage's elapsed by more than ~30s with no new
stage, the next stage is hung — and you know which one.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path


def stage(name: str, t0: float) -> float:
    """Emit a structured stage marker. Returns the new t0 for the next stage."""
    now = time.time()
    print(f"STAGE: {name} elapsed={now - t0:.2f}s", flush=True)
    return now


def build_system(kind: str):
    """Build a small, fast ASE Atoms for a given probe family."""
    if kind == "molecule":
        from ase.build import molecule

        atoms = molecule("H2O")
        # charge/spin for OMol-style models; external_field (zero — a physical
        # no-op) for polar/electrostatic models like MACE-POLAR. Models that
        # don't read a key simply ignore it.
        atoms.info = {"charge": 0, "spin": 1, "external_field": [0.0, 0.0, 0.0]}
        return atoms

    if kind == "crystal":
        from ase.build import bulk

        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
        atoms.positions[0, 0] += 0.05
        atoms.positions[1, 1] -= 0.03
        return atoms

    if kind == "slab_co":
        from ase.build import add_adsorbate, fcc111, molecule

        slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
        add_adsorbate(slab, molecule("CO"), height=2.0, position="ontop")
        return slab

    raise ValueError(f"Unknown --system kind: {kind!r}")


def load_setup(config_path: Path):
    """Import `setup` from an arbitrary Python file path."""
    spec = importlib.util.spec_from_file_location("rootstock_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "setup"):
        raise AttributeError(f"{config_path} has no setup() function")
    return module.setup


def main() -> int:
    parser = argparse.ArgumentParser(description="No-IPC probe for an MLIP config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config .py file (PEP 723 + setup function).",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint/model arg passed to setup(). Empty = setup default.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--system",
        default="molecule",
        choices=["molecule", "crystal", "slab_co"],
        help="Probe system to compute one forward pass on.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    overall_t0 = time.time()
    t0 = overall_t0
    print(
        f"PROBE: config={config_path} checkpoint={args.checkpoint!r} "
        f"device={args.device} system={args.system}",
        flush=True,
    )

    try:
        setup = load_setup(config_path)
        t0 = stage("load_setup", t0)

        atoms = build_system(args.system)
        t0 = stage(f"build_system:{args.system}:{len(atoms)}atoms", t0)

        calc = setup(args.checkpoint, args.device) if args.checkpoint else setup(device=args.device)
        t0 = stage("setup_calculator", t0)

        atoms.calc = calc
        e = atoms.get_potential_energy()
        t0 = stage(f"first_inference:E={e:.6f}eV", t0)

        f = atoms.get_forces()
        t0 = stage(f"forces:shape={f.shape}:|F|max={abs(f).max():.4f}", t0)

    except Exception:
        traceback.print_exc()
        print(f"PROBE: FAILED total_elapsed={time.time() - overall_t0:.2f}s", flush=True)
        return 1

    print(f"PROBE: OK total_elapsed={time.time() - overall_t0:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
