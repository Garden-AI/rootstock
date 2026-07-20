#!/usr/bin/env python3
"""
Numerical gate for source-built GPU extensions on ROCm.

WHY THIS EXISTS: on ROCm, torch-sparse's spmm compiles cleanly, imports fine,
runs without error - and returns WRONG NUMBERS. Its kernel is built around a
32-lane NVIDIA warp; an AMD wavefront is 64 lanes, so its shuffle-based row
reduction spans the wrong threads and drops contributions:

    dense reference  : [[2, 2, 2], [7, 7, 7]]
    torch_sparse GPU : [[2, 2, 2], [2, 2, 2]]      <-- no error raised

The probe cannot catch this. The model still produces energies and forces; they
are simply wrong. A build that succeeds is NOT evidence of a correct build  -
only comparing GPU output against CPU is.

    python3 check_numerics.py <env>

Exit 1 if any GPU op disagrees with its CPU reference.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSHOP_ROOT = Path(os.environ.get("WORKSHOP_ROOT", Path.home() / "rootstock-workshop"))
TOL = 1e-4

# Run inside the target env's interpreter, not ours.
PROGRAM = r'''
import sys
import torch

fails, checked = [], []

def cmp(name, cpu, gpu):
    err = (cpu - gpu.cpu()).abs().max().item()
    ok = err < %(tol)g
    checked.append((name, err, ok))
    if not ok:
        fails.append(name)

torch.manual_seed(0)

try:
    import torch_scatter
    src, idx = torch.randn(20000, 128), torch.randint(0, 999, (20000,))
    for nm, fn in [
        ("torch_scatter.scatter_add",  torch_scatter.scatter_add),
        ("torch_scatter.scatter_mean", torch_scatter.scatter_mean),
    ]:
        cmp(nm, fn(src, idx, dim=0, dim_size=999),
                fn(src.cuda(), idx.cuda(), dim=0, dim_size=999))
    ptr = torch.tensor([0, 100, 350, 700, 1000])
    s = torch.randn(1000, 32)
    cmp("torch_scatter.segment_csr",
        torch_scatter.segment_csr(s, ptr, reduce="sum"),
        torch_scatter.segment_csr(s.cuda(), ptr.cuda(), reduce="sum"))
except ImportError:
    pass

try:
    # The known-bad one. spmm is where the 64-lane wavefront breaks the kernel.
    from torch_sparse import SparseTensor
    D = (torch.rand(64, 64) > 0.7).float() * torch.rand(64, 64)
    r, c = D.nonzero(as_tuple=True)
    v, B = D[r, c], torch.randn(64, 8)
    cmp("torch_sparse.spmm",
        D @ B,
        SparseTensor(row=r.cuda(), col=c.cuda(), value=v.cuda(),
                     sparse_sizes=(64, 64)) @ B.cuda())
except ImportError:
    pass

try:
    import torch_cluster
    pos = torch.randn(500, 3)
    a = set(map(tuple, torch_cluster.radius_graph(pos, r=0.5).t().tolist()))
    b = set(map(tuple, torch_cluster.radius_graph(pos.cuda(), r=0.5).cpu().t().tolist()))
    checked.append(("torch_cluster.radius_graph", 0.0 if a == b else 1.0, a == b))
    if a != b:
        fails.append("torch_cluster.radius_graph")
except ImportError:
    pass

for nm, err, ok in checked:
    print("  %%-30s max|GPU-CPU| = %%.2e  %%s" %% (nm, err, "OK" if ok else "*** WRONG ***"))

if not checked:
    print("  (no source-built GPU extensions found in this env - nothing to check)")
elif fails:
    print("\nFAIL: %%d op(s) disagree with CPU: %%s" %% (len(fails), ", ".join(fails)))
    print("The build SUCCEEDED and the numbers are WRONG. Do not ship this env.")
    sys.exit(1)
else:
    print("\nPASS: every source-built GPU op matches its CPU reference.")
''' % {"tol": TOL}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    python = WORKSHOP_ROOT / "envs" / sys.argv[1] / "bin" / "python"
    if not python.exists():
        print(f"no such env: {python}")
        return 2
    print(f"numerical gate: {sys.argv[1]} (tol={TOL})")
    # cwd matters: running from a package's source checkout shadows the
    # installed module and yields a bogus ImportError.
    return subprocess.run([str(python), "-c", PROGRAM], cwd="/").returncode


if __name__ == "__main__":
    sys.exit(main())
