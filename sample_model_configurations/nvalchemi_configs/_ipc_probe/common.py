"""Shared batch construction for the IPC probe.

Imported by both bench.py (main venv) and baseline.py (worker venv), so
both sides build bit-identical inputs from the same systems .npz file.
"""

from __future__ import annotations

import numpy as np

KB_EV = 8.617333262e-5


def make_systems(kind: str, n_systems: int, target_atoms: int, seed: int = 42) -> list[dict]:
    """Build system dicts (numbers/positions/cell/pbc/velocities/charge/spin).

    kind="periodic": jittered fcc Cu supercells, trimmed to target_atoms.
    kind="molecular": a vacuum grid of jittered water molecules,
    3 * ceil(target_atoms / 3) atoms, neutral singlet.
    """
    rng = np.random.RandomState(seed)
    systems = []
    for _ in range(n_systems):
        if kind == "periodic":
            a = 3.615
            reps = max(1, round((target_atoms / 4) ** (1 / 3)))
            base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
            cells = np.stack(
                np.meshgrid(np.arange(reps), np.arange(reps), np.arange(reps), indexing="ij"),
                axis=-1,
            ).reshape(-1, 3)
            positions = (cells[:, None, :] * a + base[None, :, :]).reshape(-1, 3)
            positions = positions[:target_atoms]
            n = len(positions)
            positions = positions + rng.normal(0, 0.05, size=(n, 3))
            numbers = np.full(n, 29)  # Cu
            cell = np.eye(3) * (reps * a)
            pbc = np.array([True, True, True])
            mass = 63.546
        elif kind == "molecular":
            n_mol = max(1, target_atoms // 3)
            water = np.array([[0.0, 0.0, 0.119], [0.0, 0.763, -0.477], [0.0, -0.763, -0.477]])
            side = int(np.ceil(n_mol ** (1 / 3)))
            spacing = 3.5
            offsets = (
                np.stack(
                    np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing="ij"),
                    axis=-1,
                ).reshape(-1, 3)[:n_mol]
                * spacing
            )
            positions = (offsets[:, None, :] + water[None, :, :]).reshape(-1, 3)
            positions = positions + rng.normal(0, 0.02, size=positions.shape)
            numbers = np.tile([8, 1, 1], n_mol)
            n = len(positions)
            cell = np.eye(3) * (side * spacing + 20.0)
            pbc = np.array([False, False, False])
            mass = 6.0  # rough mean amu, only scales initial velocities
        else:
            raise ValueError(f"unknown kind {kind!r}")

        v_scale = (KB_EV * 50.0 / mass) ** 0.5
        velocities = rng.normal(0, v_scale, size=(n, 3))
        velocities -= velocities.mean(axis=0, keepdims=True)
        systems.append(
            {
                "numbers": numbers.astype(np.int64),
                "positions": positions.astype(np.float64),
                "cell": cell.astype(np.float64),
                "pbc": pbc,
                "velocities": velocities.astype(np.float64),
                "charge": np.array([0.0]),
                "spin": np.array([1.0]),
            }
        )
    return systems


def save_systems(path: str, systems: list[dict]) -> None:
    flat = {"n_systems": np.array(len(systems))}
    for i, s in enumerate(systems):
        for key, value in s.items():
            flat[f"{key}_{i}"] = value
    np.savez(path, **flat)


def load_systems(path: str) -> list[dict]:
    data = np.load(path)
    n = int(data["n_systems"])
    keys = ("numbers", "positions", "cell", "pbc", "velocities", "charge", "spin")
    return [{k: data[f"{k}_{i}"] for k in keys} for i in range(n)]


def build_batch(systems: list[dict], device: str, *, with_charge: bool, with_spin: bool):
    """Systems -> nvalchemi Batch (float32, matching AtomicData.from_atoms default)."""
    import torch
    from nvalchemi.data import AtomicData, Batch

    data_list = []
    for s in systems:
        n = len(s["numbers"])
        fields = {
            "positions": torch.tensor(s["positions"], dtype=torch.float32),
            "atomic_numbers": torch.tensor(s["numbers"], dtype=torch.long),
            "forces": torch.zeros(n, 3),
            "energy": torch.zeros(1, 1),
            "cell": torch.tensor(s["cell"], dtype=torch.float32).unsqueeze(0),
            "pbc": torch.tensor(s["pbc"], dtype=torch.bool).unsqueeze(0),
        }
        if with_charge:
            fields["charge"] = torch.tensor(s["charge"], dtype=torch.float32).reshape(1, 1)
        if with_spin:
            fields["spin"] = torch.tensor(s["spin"], dtype=torch.float32).reshape(1, 1)
        data = AtomicData(**fields)
        data.add_node_property("velocities", torch.tensor(s["velocities"], dtype=torch.float32))
        data_list.append(data)
    return Batch.from_data_list(data_list, device=torch.device(device))


def run_nve(model, batch, *, steps: int, dt: float, register_engine_nl: bool):
    """Run NVE and return (final positions, final velocities) as numpy."""
    from nvalchemi.dynamics import NVE
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.hooks import WrapPeriodicHook

    nve = NVE(model=model, dt=dt, n_steps=steps)
    if register_engine_nl:
        for hook in model.make_neighbor_hooks():
            nve.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
    if bool(batch.pbc.any()):
        nve.register_hook(WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE))
    batch = nve.run(batch)
    return (
        batch.positions.detach().cpu().numpy(),
        batch.velocities.detach().cpu().numpy(),
    )
