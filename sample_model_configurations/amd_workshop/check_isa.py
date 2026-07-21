#!/usr/bin/env python3
"""
Frontier-compatibility gate for a ROCm environment.

GPU code is compiled per-architecture. This box is gfx942 (MI300X); Frontier is
gfx90a (MI250X). A binary built only for gfx942 imports fine, passes the probe
here, and then dies on Frontier with HSA_STATUS_ERROR_INVALID_ISA / "invalid
device function" - the probe structurally CANNOT catch that. This can: it reads
the GPU code objects actually embedded in the env's binaries.

    python3 check_isa.py <env> [--require gfx90a]

Two kinds of artifact, which must be judged differently:

  * fat libraries (libtorch_hip.so, librocblas.so) embed code for MANY arches.
    Each one must itself contain the required arch.
  * per-arch kernel objects (rocblas/library/Kernels.so-000-gfx942.hsaco,
    TensileLibrary_..._gfx90a.co) are single-arch BY DESIGN - the arch is in the
    filename. Demanding gfx90a inside `...-gfx1100.hsaco` is nonsense. What
    matters is whether the *family* ships a gfx90a member alongside its gfx942
    one.

Exit 1 if a fat library or a kernel family lacks the required arch.

Verified against torch 2.9.1+rocm6.4: official ROCm wheels are fat binaries and
DO carry gfx90a. The real risk is source-built extensions (torch-scatter,
torch-sparse), which compile only for the build host's arch unless you set
    PYTORCH_ROCM_ARCH="gfx90a;gfx942"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

WORKSHOP_ROOT = Path(os.environ.get("WORKSHOP_ROOT", Path.home() / "rootstock-workshop"))

# gfx digit counts differ: gfx90a is 2 digits + letter, gfx942/gfx908 are 3,
# gfx1100 is 4. Requiring 3 digits silently never matches gfx90a - the one arch
# we care about. No capture group: findall must yield the full "gfx90a" token.
GFX = re.compile(rb"gfx\d{2,4}[a-z]?\b")

# A per-arch kernel object names its arch in the filename, e.g.
#   Kernels.so-000-gfx90a-xnack+.hsaco
#   TensileLibrary_..._CU104_gfx942.co
PER_ARCH_FILE = re.compile(r"[-_](gfx\d{2,4}[a-z]?)(-xnack[+-])?\.(hsaco|co|dat)$")

BINARY_SUFFIXES = (".so", ".hsaco", ".co", ".a")


def arches(path: Path) -> set[str]:
    """GPU architectures whose code objects are embedded in a binary."""
    try:
        blob = path.read_bytes()
    except OSError:
        return set()
    return {m.decode() for m in GFX.findall(blob)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("env", help="env name under $WORKSHOP_ROOT/envs")
    ap.add_argument("--require", default="gfx90a", help="arch Frontier needs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    root = WORKSHOP_ROOT / "envs" / args.env
    if not root.exists():
        print(f"no such env: {root}")
        return 2
    need = args.require

    fat_ok, fat_missing = [], []
    # family key -> arches shipped by that family (dir + kernel-object stem)
    families: dict[str, set[str]] = defaultdict(set)

    for path in sorted(root.rglob("*")):
        if not path.is_file() or not path.name.endswith(BINARY_SUFFIXES):
            if ".so." not in path.name:
                continue

        m = PER_ARCH_FILE.search(path.name)
        if m:
            # Single-arch by design: group into a family, judge the family later.
            stem = PER_ARCH_FILE.sub("", path.name)
            families[f"{path.parent.relative_to(root)}/{stem}"].add(m.group(1))
            continue

        found = arches(path)
        if not found:
            continue  # CPU-only binary, irrelevant
        rel = path.relative_to(root)
        (fat_ok if need in found else fat_missing).append((rel, found))

    if args.verbose:
        for rel, found in fat_ok:
            print(f"OK   {rel}  [{' '.join(sorted(found))}]")

    for rel, found in fat_missing:
        print(f"FAT-LIB MISSING {need}: {rel}  has [{' '.join(sorted(found))}]")

    # Kernel families are INFORMATIONAL, not a gate. Verified against
    # torch 2.9.1+rocm6.4: 635 of 914 families have no gfx90a member, and they
    # are overwhelmingly FP8 (B8B8/F8) GEMM variants plus all of hipsparselt.
    # That is not a packaging defect - MI250X has no FP8 units and no structured
    # -sparsity engine, so AMD correctly ships no gfx90a kernels for them. Gating
    # on this would fail every environment forever. What actually decides whether
    # torch loads and runs on Frontier is the fat libraries.
    fam_missing = {k: v for k, v in families.items() if need not in v}
    fp8_ish = sum(1 for k in fam_missing if re.search(r"B8|F8|XF32", k))
    sparse = sum(1 for k in fam_missing if "hipsparselt" in k)

    print(
        f"fat libraries : {len(fat_ok)} carry {need}, {len(fat_missing)} do not  <-- the gate\n"
        f"kernel families: {len(families) - len(fam_missing)} carry {need}, "
        f"{len(fam_missing)} do not (informational)\n"
        f"  of those, {fp8_ish} are FP8/XF32 variants and {sparse} are hipsparselt  - \n"
        f"  datatypes/engines MI250X physically lacks, so their absence is expected."
    )

    if fat_missing:
        print(
            f"\nFAIL: the fat libraries above lack {need} and would hit\n"
            f"HSA_STATUS_ERROR_INVALID_ISA on Frontier. If these are source-built\n"
            f'extensions (torch-scatter/torch-sparse), rebuild with\n'
            f'PYTORCH_ROCM_ARCH="gfx90a;gfx942".'
        )
        return 1
    print(f"\nPASS: every fat GPU library in '{args.env}' carries {need} - Frontier-compatible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
