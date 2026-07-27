# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "sevenn>=0.10.0",
#     "ase>=3.22",
#     "torch>=2.0",
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
"""SevenNet env — hosts pretrained SevenNet universal potentials.

SevenNet (SCalable EquiVariance-Enabled Neural Network) ships several
pretrained models, loaded by keyword through ``SevenNetCalculator``.

Multi-fidelity models (``7net-omni``, ``7net-mf-ompa``) take a ``modal``
argument selecting the training fidelity (e.g. ``"mpa"`` or ``"omat24"``),
and sevenn releases after mid-2026 make it mandatory (older ones defaulted
silently). When not given, setup() defaults them to ``"mpa"`` — the
MPtrj+sAlex fidelity, consistent with the lineage of the other checkpoints
here — so no-kwarg paths (smoke-test, plain ``rootstock add``) keep working.
Override at runtime via ``setup_kwargs={"modal": ...}`` on
RootstockCalculator (or ``--kwarg modal=...`` for ``rootstock add``);
single-fidelity models ignore it.
"""

CHECKPOINTS = {
    "sevennet-0": "7net-0",
    "sevennet-l3i5": "7net-l3i5",
    "sevennet-omat": "7net-omat",
    "sevennet-mf-ompa": "7net-mf-ompa",
    "sevennet-omni": "7net-omni",
    # Your own fine-tuned weights: pair with weights= (loaded via setup_from_path).
    "sevennet:custom": None,
}


# Multi-fidelity models and the fidelity used when modal isn't specified.
MULTI_FIDELITY_DEFAULT_MODAL = {
    "sevennet-mf-ompa": "mpa",
    "sevennet-omni": "mpa",
}


def setup(checkpoint: str, device: str = "cuda", modal: str | None = None):
    """
    Load a SevenNet calculator.

    Args:
        checkpoint: Canonical checkpoint id, must be a key of CHECKPOINTS.
        device: PyTorch device string (e.g., "cuda", "cuda:0", "cpu").
        modal: Fidelity selector for multi-fidelity models (e.g. "mpa",
            "omat24"). Defaults to "mpa" for 7net-omni / 7net-mf-ompa;
            ignored by the single-fidelity models.

    Returns:
        ASE-compatible calculator.
    """
    from sevenn.calculator import SevenNetCalculator

    if modal is None:
        modal = MULTI_FIDELITY_DEFAULT_MODAL.get(checkpoint)
    kwargs = {"modal": modal} if modal is not None else {}
    return SevenNetCalculator(model=CHECKPOINTS[checkpoint], device=device, **kwargs)


def setup_from_path(path: str, device: str = "cuda", modal: str | None = None):
    # Local checkpoints (`rootstock add-local`): SevenNetCalculator loads a
    # checkpoint file directly. A multi-fidelity fine-tune must register its
    # fidelity (--kwarg modal=mpa); single-fidelity ones need no kwargs.
    from sevenn.calculator import SevenNetCalculator

    kwargs = {"modal": modal} if modal is not None else {}
    return SevenNetCalculator(model=path, device=device, **kwargs)
