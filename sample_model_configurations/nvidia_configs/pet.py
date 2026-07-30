# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#     "upet>=0.2.6",
#     "ase>=3.22",
#     # upet pulls nvalchemi-toolkit-ops unpinned; 0.4+ needs torch>=2.8 at
#     # runtime but only declares the constraint on its extras, so the
#     # resolver won't catch it (same trap as the tensornet env).
#     "torch>=2.8,<2.14",
# ]
# ///
"""PET env — hosts lab-cosmo's UPET foundation checkpoints (PET-MAD successor).

The upstream string encodes model@version; versions are pinned rather than
"latest" so rebuilds serve the same weights. pet-omatpes-l is trained at the
r2SCAN level of theory — its energies are not comparable to the PBE-level
pet-oam models. Weights download ungated from the lab-cosmo/upet HF repo.
"""

CHECKPOINTS = {
    "pet-oam-xl": "pet-oam-xl@1.0.0",
    "pet-omatpes-l": "pet-omatpes-l@0.1.0",
}


def setup(checkpoint: str, device: str = "cuda"):
    from upet.calculator import UPETCalculator

    model, version = CHECKPOINTS[checkpoint].split("@", 1)
    return UPETCalculator(model=model, version=version, device=device)
