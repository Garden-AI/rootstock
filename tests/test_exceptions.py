"""One base class catches every rootstock domain error."""

from __future__ import annotations

import pytest

from rootstock import RootstockError
from rootstock.environment import CheckpointNotFoundError
from rootstock.manifest import ManifestError
from rootstock.operations import OperationError
from rootstock.server import WorkerDiedError


@pytest.mark.parametrize(
    "exc", [CheckpointNotFoundError, WorkerDiedError, OperationError, ManifestError]
)
def test_domain_errors_share_the_base(exc):
    assert issubclass(exc, RootstockError)


def test_historic_stdlib_categories_preserved():
    """Existing `except LookupError` / `except RuntimeError` call sites must
    keep working across the taxonomy change."""
    assert issubclass(CheckpointNotFoundError, LookupError)
    assert issubclass(WorkerDiedError, RuntimeError)
    assert issubclass(OperationError, RuntimeError)
    assert issubclass(ManifestError, RuntimeError)


def test_catching_the_base_catches_a_raise():
    with pytest.raises(RootstockError):
        raise CheckpointNotFoundError("no env declares 'nope'")
