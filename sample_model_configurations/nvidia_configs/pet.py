# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#     "upet>=0.2.6",
#     "ase>=3.22",
#     # upet pulls nvalchemi-toolkit-ops unpinned; 0.4+ needs torch>=2.8 at
#     # runtime but only declares the constraint on its extras, so the
#     # resolver won't catch it (the same trap that bit the tensornet env).
#     "torch>=2.8,<2.14",
# ]
# ///
"""PET env — hosts lab-cosmo's UPET foundation checkpoints (PET-MAD successor).

The upstream string encodes model@version; versions are pinned rather than
"latest" so rebuilds serve the same weights. pet-omatpes-l is trained at the
r2SCAN level of theory — its energies are not comparable to the PBE-level
pet-oam models.
"""

CHECKPOINTS = {
    "pet-oam-xl": "pet-oam-xl@1.0.0",
    "pet-omatpes-l": "pet-omatpes-l@0.1.0",
}


def setup(checkpoint: str, device: str = "cuda", **kwargs):
    from huggingface_hub import hf_hub_download
    from upet.calculator import UPETCalculator

    # Force metatomic's vesin neighbor-list fallback. metatomic-ase 0.1.2's
    # nvalchemi fast path passes max_neighbors = len(system) * max(128,
    # cutoff**3) — a float whenever the cutoff exceeds 128**(1/3) ≈ 5.04 Å —
    # into nvalchemiops' torch.full size tuple, which raises TypeError. The
    # path auto-activates on CUDA because upet hard-depends on
    # nvalchemi-toolkit-ops, so the import always succeeds. The flag is read
    # at call time, so patching after import works.
    import metatomic_ase._neighbors as _neighbors

    _neighbors.HAS_NVALCHEMIOPS = False

    # Passing model=/version= makes UPETCalculator resolve the name by listing
    # the hub repo — an uncached API call that fails on workers, which run
    # with HF_HUB_OFFLINE=1 (and on any node without internet). Fetch the
    # pinned file ourselves — a cache hit needs no network even offline — and
    # hand it over as checkpoint_path, which skips the resolve entirely.
    model, version = CHECKPOINTS[checkpoint].split("@", 1)
    path = hf_hub_download(
        repo_id="lab-cosmo/upet",
        filename=f"{model}-v{version}.ckpt",
        subfolder="models",
    )
    return UPETCalculator(checkpoint_path=path, device=device, **kwargs)
