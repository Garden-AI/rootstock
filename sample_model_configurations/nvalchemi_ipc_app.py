"""
Modal harness for the batched (nvalchemi) IPC prototype.

Each family gets one image holding the production two-venv topology in a
single container: the *main* interpreter runs the nvalchemi engine plus
the rootstock client (``RootstockModel``), and a hand-built venv at
``/rs-root/envs/<family>`` plays the pre-built Rootstock environment
(nvalchemi + the family's model stack + the rootstock wheel, so the
spawned batched worker imports match production). The probe measures
correctness, NVE parity, and IPC overhead against an in-process baseline
run inside the worker venv.

Usage:
    modal run nvalchemi_ipc_app.py::probe_mace
    modal run nvalchemi_ipc_app.py::probe_mace_cpu --mode correctness
    modal run nvalchemi_ipc_app.py::probe_mace --neighbor-mode engine
    modal run nvalchemi_ipc_app.py::probe_uma --task omol
    modal run nvalchemi_ipc_app.py::probe_aimnet2
"""

import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("rootstock-nvalchemi-ipc")

HERE = Path(__file__).parent
CONFIGS = HERE / "nvalchemi_configs"
PROBE = CONFIGS / "_ipc_probe"
REPO = HERE.parent

ROOT = "/rs-root"
model_cache = modal.Volume.from_name("rootstock-model-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")

# Engine-side stack: nvalchemi + its GPU ops (NeighborListHook, integrator
# kernels). The model families themselves live only in the worker venvs.
MAIN_DEPS = [
    "nvalchemi-toolkit",
    "nvalchemi-toolkit-ops[torch-cu12]>=0.4.1",
    "ase>=3.22",
]


def _build_wheel() -> Path:
    """Build the current working tree into a wheel (uncommitted changes included)."""
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(REPO / "dist")],
        check=True,
        cwd=REPO,
        stdout=sys.stderr,
    )
    wheels = sorted((REPO / "dist").glob("rootstock-*.whl"), key=lambda p: p.stat().st_mtime)
    return wheels[-1]


# In containers the image is already built; the placeholder only keeps
# module-level image construction importable there.
WHEEL = _build_wheel() if modal.is_local() else Path("rootstock-0-py3-none-any.whl")
WHEEL_IMG = f"/tmp/{WHEEL.name}"


def ipc_image(family: str, worker_deps: list[str]) -> modal.Image:
    venv_python = f"{ROOT}/envs/{family}/bin/python"
    quoted = " ".join(f"'{d}'" for d in worker_deps)
    return (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(*MAIN_DEPS)
        .add_local_file(str(WHEEL), WHEEL_IMG, copy=True)
        .run_commands(
            f"/.uv/uv pip install --system {WHEEL_IMG}",
            f"/.uv/uv venv {ROOT}/envs/{family} --python 3.11",
            f"/.uv/uv pip install --python {venv_python} {quoted} {WHEEL_IMG}",
        )
        .add_local_file(
            str(CONFIGS / f"{family}.py"), f"{ROOT}/envs/{family}/env_source.py", copy=True
        )
        .add_local_file(str(PROBE / "bench.py"), "/rs-probe/bench.py")
        .add_local_file(str(PROBE / "baseline.py"), "/rs-probe/baseline.py")
        .add_local_file(str(PROBE / "common.py"), "/rs-probe/common.py")
        .add_local_file(str(PROBE / "uma_diag.py"), "/rs-probe/uma_diag.py")
    )


def _run_probe(family: str, checkpoint: str, **kwargs) -> int:
    cmd = [
        sys.executable,
        "/rs-probe/bench.py",
        "--family",
        family,
        "--root",
        ROOT,
        "--env",
        family,
        "--checkpoint",
        checkpoint,
    ]
    for key, value in kwargs.items():
        if value is not None:
            cmd += [f"--{key.replace('_', '-')}", str(value)]
    print(f"RUN: {' '.join(cmd)}", flush=True)
    import os

    env = dict(os.environ)
    # Expandable segments fight OOM from three torch processes on one GPU,
    # but cudaIpcGetMemHandle cannot export them — CUDA-transport runs go
    # without.
    if kwargs.get("transport") != "cuda":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.run(cmd, env=env)
    model_cache.commit()
    return proc.returncode


mace_image = ipc_image(
    "mace",
    ["nvalchemi-toolkit[mace]", "nvalchemi-toolkit-ops[torch-cu12]>=0.4.1"],
)


@app.function(
    image=mace_image,
    gpu="A10G",
    volumes={f"{ROOT}/cache": model_cache},
    secrets=[HF_SECRET],
    timeout=3600,
)
def probe_mace(
    mode: str = "all",
    neighbor_mode: str = "worker",
    checkpoint: str = "mace-medium-0b2-batched",
    grid: str | None = None,
    dtype: str | None = None,
    transport: str = "socket",
):
    assert (
        _run_probe(
            "mace",
            checkpoint,
            mode=mode,
            neighbor_mode=neighbor_mode,
            device="cuda",
            grid=grid,
            setup_kwargs=f'{{"dtype": "{dtype}"}}' if dtype else None,
            transport=transport,
        )
        == 0
    )


@app.function(
    image=mace_image,
    volumes={f"{ROOT}/cache": model_cache},
    secrets=[HF_SECRET],
    timeout=3600,
    cpu=8.0,
)
def probe_mace_cpu(
    mode: str = "correctness",
    neighbor_mode: str = "worker",
    checkpoint: str = "mace-medium-0b2-batched",
):
    assert (
        _run_probe(
            "mace",
            checkpoint,
            mode=mode,
            neighbor_mode=neighbor_mode,
            device="cpu",
            grid="1x32,4x32",
            iters=5,
            nve_steps=20,
        )
        == 0
    )


uma_image = ipc_image("uma", ["nvalchemi-toolkit[uma]"])


@app.function(
    image=uma_image,
    gpu="A10G",
    volumes={f"{ROOT}/cache": model_cache},
    secrets=[HF_SECRET],
    timeout=3600,
)
def probe_uma(
    mode: str = "all",
    task: str = "omol",
    checkpoint: str = "uma-s-1p1-batched",
    grid: str | None = None,
):
    assert (
        _run_probe(
            "uma",
            checkpoint,
            mode=mode,
            neighbor_mode="worker",
            device="cuda",
            setup_kwargs=f'{{"task": "{task}"}}',
            grid=grid,
        )
        == 0
    )


@app.function(
    image=uma_image,
    gpu="A10G",
    volumes={f"{ROOT}/cache": model_cache},
    secrets=[HF_SECRET],
    timeout=3600,
    # The MD-drift regime provokes repeated dynamo recompiles of fairchem's
    # compiled path, which are host-RAM-hungry; the default request gets the
    # container evicted mid-diag.
    memory=32768,
)
def probe_uma_diag():
    import os

    from rootstock.environment import get_model_cache_env

    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.update(get_model_cache_env(Path(ROOT)))
    proc = subprocess.run(
        [
            f"{ROOT}/envs/uma/bin/python",
            "/rs-probe/uma_diag.py",
            f"{ROOT}/envs/uma",
            "cuda",
        ],
        env=env,
    )
    model_cache.commit()
    assert proc.returncode == 0


aimnet2_image = ipc_image(
    "aimnet2",
    ["nvalchemi-toolkit[aimnet]", "nvalchemi-toolkit-ops[torch-cu12]>=0.4.1"],
)


@app.function(
    image=aimnet2_image,
    gpu="A10G",
    volumes={f"{ROOT}/cache": model_cache},
    secrets=[HF_SECRET],
    timeout=3600,
)
def probe_aimnet2(
    mode: str = "all",
    neighbor_mode: str = "worker",
    checkpoint: str = "aimnet2-batched",
    grid: str | None = None,
    transport: str = "socket",
):
    assert (
        _run_probe(
            "aimnet2",
            checkpoint,
            mode=mode,
            neighbor_mode=neighbor_mode,
            device="cuda",
            grid=grid,
            transport=transport,
        )
        == 0
    )
