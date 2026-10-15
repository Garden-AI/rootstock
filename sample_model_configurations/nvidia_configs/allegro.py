# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # setup() uses nequip's loader internals and keys its compile cache on the
#     # nequip version, so stay within one minor release.
#     "nequip>=0.19,<0.20",
#     "nequip-allegro>=0.8.3,<0.9",
#     "ase>=3.23",
#     # AOTInductor compilation needs torch>=2.6. Listed here so the explicit
#     # cu128 index below routes it: PyPI's recent torch wheels bundle a CUDA
#     # newer than many cluster drivers support.
#     "torch>=2.6,<2.12",
#     # CUDA headers for compiling on nodes with no CUDA toolkit loaded (the
#     # runtime headers come with torch); matches the cu128 torch build.
#     "cuda-toolkit[cccl,nvcc]>=12.8,<12.9",
#     # AOTInductor's C++ builder imports torch.utils.cpp_extension, which
#     # imports setuptools at module level; uv venvs do not seed it.
#     "setuptools>=70",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Allegro env — hosts the Allegro foundation models from nequip.net.

Allegro is a strictly local E(3)-equivariant interatomic potential from the
Kozinsky group (Harvard), built on the NequIP framework; its locality is what
lets it scale across many GPUs in LAMMPS. Allegro-OAM-L is trained on OMat24
and fine-tuned on sAlex + MPtrj; Allegro-MP-L is trained on MPtrj alone. All
are PBE-level and licensed CC-BY-4.0. Versions are pinned to fixed Zenodo
artifacts (md5-checked) so rebuilds serve the same weights; the package file
is downloaded into the shared model cache at `rootstock add` time and loaded
from there, so serving needs no network.

Compilation: on GPU, models run through NequIP's AOTInductor path
(``nequip-compile --mode aotinductor --target ase``), the path NequIP
recommends for inference. A compiled model only runs on the hardware and
software it was compiled with, so setup() compiles once per (model, GPU
architecture, torch version, nequip version) and caches the result: in the
shared model cache when the caller can write there, otherwise in the caller's
TorchInductor cache directory. The first GPU load of a model on a new GPU type
therefore compiles first: on an 8-core node with an A10G, about 2 minutes for
Allegro-OAM-L, 5 for NequIP-OAM-L and 6-7 for NequIP-OAM-XL. If that load runs
into the worker startup timeout, raise it (``timeout=`` on
RootstockCalculator, ``--verify-timeout`` for verification); an interrupted
compile still finishes in the background and later loads reuse the compiled
file. AOTInductor needs a C++17 host compiler (``c++``/``gcc`` >= 8, or
whatever ``CXX`` names) at compile time; CUDA headers come from the env's own
NVIDIA wheels when no CUDA toolkit is loaded. Pass
``setup_kwargs={"compile": False}`` to run eager instead. CPU loads always run
eager.

Custom weights: ``allegro:custom`` with weights= accepts a packaged model
(``.nequip.zip``), a training checkpoint (``.ckpt``), or a model already
compiled for ASE on this hardware (``.nequip.pt2``). If your model's type
names are not chemical symbols, pass
``setup_kwargs={"chemical_species_to_atom_type_map": {"H": "<type>", ...}}``.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "allegro-oam-l": "Allegro-OAM-L-0.1",
    "allegro-mp-l": "Allegro-MP-L-0.1",
    # Your own Allegro model: pair with weights= (loaded via setup_from_path).
    "allegro:custom": None,
}

# Zenodo record and md5 of each pinned package file (the artifacts nequip.net
# resolves mir-group/<model>:0.1 to).
_ARTIFACTS = {
    "Allegro-OAM-L-0.1": ("16980200", "0db7f9b3c3a62e74d78b3fcf2973c462"),
    "Allegro-MP-L-0.1": ("16980200", "c68d5345e925ee444543d6d4d968161f"),
}


def _shared_cache() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nequip"


def _package_path(name: str) -> Path:
    """The pinned package file in the shared cache, downloading it if missing."""
    path = _shared_cache() / "packages" / f"{name}.nequip.zip"
    if path.is_file():
        return path
    record, md5 = _ARTIFACTS[name]
    url = f"https://zenodo.org/api/records/{record}/files/{name}.nequip.zip/content"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        digest = hashlib.md5()
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != md5:
            raise RuntimeError(f"{url}: md5 {digest.hexdigest()} does not match {md5}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _compiled_dirs() -> list[Path]:
    """Where compiled models are looked up, in order; the first writable one
    receives new compilations."""
    dirs = [_shared_cache() / "compiled"]
    inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if inductor:
        dirs.append(Path(inductor) / "nequip")
    else:
        dirs.append(Path.home() / ".cache" / "nequip" / "compiled")
    return dirs


def _writable(directory: Path) -> bool:
    probe = directory
    while not probe.exists():
        probe = probe.parent
    return os.access(probe, os.W_OK | os.X_OK)


def _hardware_tag(device: str) -> str:
    import torch

    major, minor = torch.cuda.get_device_capability(torch.device(device))
    return f"sm{major}{minor}"


def _toolkit_env(env: dict, link_dir: Path) -> None:
    """Point AOTInductor at CUDA headers and libcuda when no toolkit is loaded.

    The C++ wrapper AOTInductor builds needs ``$CUDA_HOME/include`` and links
    ``-lcuda``. Compute nodes often expose a CUDA toolkit only through a
    module, so when none is visible fall back to the headers in NVIDIA's CUDA
    wheels (``nvidia/cuda_runtime/``, ``cuda_nvcc/`` and ``cuda_cccl/``), and
    link against the driver library this process has loaded.
    """
    if env.get("CUDA_HOME") or env.get("CUDA_PATH") or shutil.which("nvcc", path=env.get("PATH")):
        return
    import nvidia

    headers = next(
        (h for base in nvidia.__path__ for h in Path(base).glob("*/include/cuda.h")), None
    )
    if headers is None:
        return
    env["CUDA_HOME"] = str(headers.parent.parent)
    extra = [
        str(h.parent.parent)
        for base in nvidia.__path__
        for marker in ("crt/host_defines.h", "nv/target")
        for h in Path(base).glob(f"*/include/{marker}")
    ]
    env["CPATH"] = os.pathsep.join(filter(None, [*extra, env.get("CPATH")]))
    with open("/proc/self/maps") as maps:
        libcuda = next((line.split()[-1] for line in maps if "/libcuda.so" in line), None)
    if libcuda is not None:
        (link_dir / "libcuda.so").symlink_to(libcuda)
        env["LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(link_dir), env.get("LIBRARY_PATH")])
        )


def _compiled_model(source: Path, key: str, device: str) -> Path:
    """Path to ``source`` compiled for ASE on ``device``, compiling on a miss."""
    import nequip
    import torch

    tag = f"{key}-{_hardware_tag(device)}-torch{torch.__version__}-nequip{nequip.__version__}"
    name = f"{tag.replace('+', '_')}.nequip.pt2"
    dirs = _compiled_dirs()
    for directory in dirs:
        if (directory / name).is_file():
            return directory / name

    target_dir = next((d for d in dirs if _writable(d)), None)
    if target_dir is None:
        raise RuntimeError(f"no writable directory to cache the compiled model in: {dirs}")
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / name
    tmp = target_dir / f".{tag}.{os.getpid()}.nequip.pt2"

    # Compile in a child process pinned to the same GPU, so the compiler's
    # memory is released before the model is loaded. The child moves its
    # output into place itself: if this worker is killed mid-compile (e.g. by
    # a startup timeout), the compile still completes and lands in the cache.
    env = dict(os.environ)
    if ":" in device:
        index = int(device.split(":", 1)[1])
        visible = env.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
    compile_and_move = (
        "import os, sys; from nequip.scripts.compile import main; "
        "main(sys.argv[1:-1]); os.replace(sys.argv[2], sys.argv[-1])"
    )
    cmd = [
        sys.executable,
        "-c",
        compile_and_move,
        str(source),
        str(tmp),
        "--mode",
        "aotinductor",
        "--device",
        "cuda",
        "--target",
        "ase",
        str(out),
    ]
    print(f"nequip: compiling {source.name} for {tag} (one-time, several minutes)", file=sys.stderr)
    try:
        with tempfile.TemporaryDirectory() as link_dir:
            _toolkit_env(env, Path(link_dir))
            result = subprocess.run(cmd, env=env, stdout=sys.stderr, stderr=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"nequip-compile failed (exit {result.returncode}); see the log above. "
                'Pass setup_kwargs={"compile": False} to run eager instead.'
            )
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _load(source: Path, key: str, device: str, compile: bool, kwargs: dict):
    # torch's AOTInductor package loader reaches torch._inductor.codecache
    # without importing it, which fails in a fresh process.
    import torch._inductor.codecache  # noqa: F401
    from nequip.integrations.ase import NequIPCalculator

    if source.name.endswith(".nequip.pt2"):
        return NequIPCalculator.from_compiled_model(str(source), device=device, **kwargs)
    if compile and device != "cpu":
        compiled = _compiled_model(source, key, device)
        return NequIPCalculator.from_compiled_model(str(compiled), device=device, **kwargs)
    # Eager load of a package or checkpoint. nequip exposes this only as a
    # private classmethod (its own tests use it); the public loader takes
    # compiled models only.
    return NequIPCalculator._from_saved_model(str(source), device=device, **kwargs)


def setup(checkpoint: str, device: str = "cuda", compile: bool = True, **kwargs):
    """
    Load an Allegro calculator for a pinned nequip.net foundation model.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu").
        compile: Run the AOTInductor-compiled model on GPU (compiling and
            caching it on first use). False runs eager PyTorch.

    Returns:
        ASE-compatible calculator.
    """
    name = CHECKPOINTS[checkpoint]
    kwargs.setdefault("chemical_species_to_atom_type_map", True)
    return _load(_package_path(name), name, device, compile, kwargs)


def setup_from_path(path: str, device: str = "cuda", compile: bool = True, **kwargs):
    # Custom checkpoints (`:custom` ids with user weights): an Allegro .nequip.zip
    # package or .ckpt checkpoint (compiled on first GPU use, cached under a
    # key that includes the file's hash) or a ready .nequip.pt2.
    source = Path(path)
    key = source.name
    if not key.endswith(".nequip.pt2") and compile and device != "cpu":
        digest = hashlib.sha256()
        with open(source, "rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        key = f"custom-{digest.hexdigest()[:16]}"
    return _load(source, key, device, compile, kwargs)
