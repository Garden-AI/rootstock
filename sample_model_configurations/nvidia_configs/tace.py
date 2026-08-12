# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tace",
#     "ase>=3.22",
#     "torch>=2.4,<2.14",
# ]
#
# # The TECE architecture and its foundation registry ship in tace 0.2.0,
# # which is not on PyPI (latest release there is 0.1.0). This is the commit
# # the Matbench Discovery submission pinned.
# [tool.uv.sources.tace]
# git = "https://github.com/xvzemin/tace"
# rev = "81f65a4c188bd09cec8d1419388f7afdcc1b6fd0"
# ///
"""TACE env — hosts TECE/TACE foundation checkpoints (Xu, Xie & Hu).

openequivariance CUDA-kernel acceleration is optional upstream and not
installed here — TACE runs on the e3nn path.
"""

CHECKPOINTS = {
    "tece-oam-rra-1.0": "TECE-OAM-RRA-1.0",
}


def setup(checkpoint: str, device: str = "cuda", **kwargs):
    from tace.foundations import tace_foundations
    from tace.interface.ase import TACEAseCalc

    model_path = tace_foundations[CHECKPOINTS[checkpoint]]
    kwargs.setdefault("dtype", "float32")
    return TACEAseCalc(model_path, device=device, **kwargs)
