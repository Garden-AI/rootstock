"""
Batched serving: expose a Rootstock-hosted MLIP as an nvalchemi model.

``RootstockModel`` is a ``BaseModelMixin``-compatible proxy that runs the
real nvalchemi model wrapper (MACEWrapper, UMAWrapper, ...) inside a
pre-built Rootstock environment in a worker subprocess, and forwards
batches over a Unix socket using the wire protocol in ``wire.py``. The
main process needs nvalchemi installed (it hosts the dynamics engine);
the worker environment carries the model family's conflicting stack.

Import ``RootstockModel`` lazily — the client package must stay importable
without torch/nvalchemi:

    from rootstock.batched import RootstockModel
"""


def __getattr__(name):
    if name == "RootstockModel":
        from rootstock.batched.model import RootstockModel

        return RootstockModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
