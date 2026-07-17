"""Rootstock's exception taxonomy.

One base class so callers can catch every rootstock domain error with a
single ``except RootstockError``. The concrete exceptions live next to the
code that raises them and inherit both this base and the stdlib category
they historically subclassed, so existing ``except LookupError`` /
``except RuntimeError`` call sites keep working:

- ``CheckpointNotFoundError`` (rootstock.environment) — no installed env
  declares the requested canonical checkpoint id.
- ``WorkerDiedError`` (rootstock.server) — the worker process failed at the
  socket level; carries a post-mortem.
- ``OperationError`` (rootstock.operations) — an install/add operation
  failed; message is user-presentable.
- ``ManifestError`` (rootstock.manifest) — the manifest exists but cannot
  be used; deliberately not treated as "no manifest".
"""

from __future__ import annotations


class RootstockError(Exception):
    """Base class for all rootstock domain errors."""
