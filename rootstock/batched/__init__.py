"""
Batched serving: expose a Rootstock-hosted MLIP as an nvalchemi model.

``RootstockModel`` is a ``BaseModelMixin``-compatible proxy that runs the
real nvalchemi model wrapper (MACEWrapper, UMAWrapper, ...) inside a
pre-built Rootstock environment in a worker subprocess, and forwards
batches over a Unix socket using the wire protocol in ``wire.py``. The
main process needs nvalchemi installed (it hosts the dynamics engine);
the worker environment carries the model family's conflicting stack.

Import ``AlchemiModel`` lazily — the client package must stay importable
without torch/nvalchemi:

    from rootstock.batched import AlchemiModel
"""


def __getattr__(name):
    if name == "AlchemiModel":
        from rootstock.batched.model import AlchemiModel

        return AlchemiModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
