#!/usr/bin/env python3
"""
AMD workshop for crafting MLIP configurations on ROCm, targeting Frontier.

The ROCm counterpart of ../modal_app.py. Same philosophy: this is a workshop,
not a validator - the artifact we ship is the config file in ../amd_configs/.
But instead of Modal images we build one uv venv per config on the box (the
same thing `rootstock install` does on HPC), and instead of `modal run` we
just run ../_agent/probe.py in that venv. STAGE markers stream to stdout, so
running this over ssh gives the same realtime observability as Modal did.

A green probe is necessary but NOT sufficient for Frontier. Also run:
    check_isa.py <env> --require gfx90a   # is Frontier's arch even in the binary?
    check_numerics.py <env>               # do source-built GPU ops match CPU?

Runs ON the AMD box (stdlib only, needs python3 >= 3.8 and uv on PATH):

    python3 workshop.py probe ../amd_configs/mace.py
    python3 workshop.py probe ../amd_configs/mace.py --checkpoint mace-off23-medium --system molecule
    python3 workshop.py probe ../amd_configs/uma.py --fresh   # rebuild the venv
    python3 workshop.py list

Environment resolution honors the config's full PEP 723 metadata, including
[tool.uv.index] blocks (this is how the ROCm torch index is selected):
`uv export --script` resolves the script's deps to a lockfile, which is then
installed into a plain venv. Iteration is fast because uv's wheel cache plays
the role Modal image layers did.

Weight caches live under WORKSHOP_ROOT/cache with the same env redirection
modal_app.py used (HOME, XDG_CACHE_HOME, HF_HOME, ...), mirroring what
rootstock does on HPC. Set HF_TOKEN in your shell for gated checkpoints.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE.parent / "_agent" / "probe.py"

WORKSHOP_ROOT = Path(os.environ.get("WORKSHOP_ROOT", Path.home() / "rootstock-workshop"))
ENVS = WORKSHOP_ROOT / "envs"
CACHE = WORKSHOP_ROOT / "cache"
PROBE_TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "900"))

CACHE_ENV = {
    "HOME": str(CACHE / "home"),
    "XDG_CACHE_HOME": str(CACHE),
    "HF_HOME": str(CACHE / "huggingface"),
    "HF_HUB_CACHE": str(CACHE / "huggingface" / "hub"),
    "HF_XET_CACHE": str(CACHE / "huggingface" / "xet"),
}


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"RUN: {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def requires_python(config: Path) -> str | None:
    """Pull requires-python out of the PEP 723 block, e.g. '>=3.11' -> '3.11'."""
    m = re.search(r'^#\s*requires-python\s*=\s*"[>=~]*(\d+\.\d+)', config.read_text(), re.M)
    return m.group(1) if m else None


def index_urls(config: Path) -> list[str]:
    """Index URLs declared in the config's [[tool.uv.index]] blocks.

    `uv export --script` reads these from the PEP 723 metadata, but the
    resulting lockfile only carries pinned versions - `uv pip install -r` would
    then look for e.g. pytorch-triton-rocm (ROCm-index-only) on PyPI and fail.
    So the same indexes are handed to the install step explicitly.
    """
    return re.findall(r'^#\s*url\s*=\s*"([^"]+)"', config.read_text(), re.M)


def build_env(config: Path, fresh: bool) -> Path:
    """Create (or reuse) a venv for the config from its PEP 723 metadata."""
    name = config.stem
    venv = ENVS / name
    lock = ENVS / f"{name}.lock.txt"
    python = venv / "bin" / "python"

    if fresh and venv.exists():
        sh(["rm", "-rf", str(venv)])

    # Resolve script deps honoring [tool.uv.index] etc. Re-export every time  -
    # it's cheap and catches config edits; the install below is a no-op when
    # nothing changed.
    r = sh(["uv", "export", "--script", config, "--no-hashes", "-o", lock])
    if r.returncode != 0:
        raise SystemExit(f"uv export failed for {config}")

    if not python.exists():
        cmd = ["uv", "venv", venv]
        py = requires_python(config)
        if py:
            cmd += ["--python", py]
        if sh(cmd).returncode != 0:
            raise SystemExit(f"uv venv failed for {name}")

    cmd = ["uv", "pip", "install", "--python", python, "-r", lock]
    for url in index_urls(config):
        cmd += ["--index", url]
    if index_urls(config):
        # The lock pins exact versions (already resolved under the config's
        # `explicit` index rules), but by default uv takes each package from the
        # *first* index that carries it at all - so a package the ROCm index
        # happens to mirror at an older version (filelock, sympy, ...) fails to
        # match the pin. Searching all indexes is safe against a pinned lock.
        cmd += ["--index-strategy", "unsafe-best-match"]
    r = sh(cmd)
    if r.returncode != 0:
        raise SystemExit(f"uv pip install failed for {name} - grep _agent/failure_modes.md")
    return python


def cmd_probe(args) -> int:
    config = Path(args.config).resolve()
    if not config.exists():
        raise SystemExit(f"no such config: {config}")
    ENVS.mkdir(parents=True, exist_ok=True)
    for p in CACHE_ENV.values():
        os.makedirs(p, exist_ok=True)

    python = build_env(config, fresh=args.fresh)

    cmd = [python, PROBE, "--config", config, "--system", args.system, "--device", args.device]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    env = {**os.environ, **CACHE_ENV}
    print(f"PROBE_CMD: {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        return sh(cmd, env=env, timeout=PROBE_TIMEOUT).returncode
    except subprocess.TimeoutExpired:
        print(f"PROBE: TIMEOUT after {PROBE_TIMEOUT}s - last STAGE marker names the hung stage")
        return 124


def cmd_list(_args) -> int:
    if not ENVS.exists():
        print("(no envs built yet)")
        return 0
    for d in sorted(ENVS.iterdir()):
        if (d / "bin" / "python").exists():
            print(d.name)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="build/reuse the config's venv and run one probe")
    pr.add_argument("config", help="path to an amd_configs/<name>.py file")
    pr.add_argument("--checkpoint", default="", help="canonical checkpoint id (empty = setup default)")
    pr.add_argument("--system", default="crystal", choices=["molecule", "crystal", "slab_co"])
    pr.add_argument("--device", default="cuda", help="'cuda' also selects AMD GPUs under ROCm torch")
    pr.add_argument("--fresh", action="store_true", help="delete and rebuild the venv first")
    pr.set_defaults(fn=cmd_probe)

    ls = sub.add_parser("list", help="list built envs")
    ls.set_defaults(fn=cmd_list)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
