"""Server-side parsing of in-band worker errors from FORCEREADY extras."""

from __future__ import annotations

import json

import pytest

from rootstock.server import _worker_error_from_extra


def test_error_payload_is_extracted():
    tb = 'Traceback (most recent call last):\n  ...\nValueError("boom")'
    extra = json.dumps({"error": tb}).encode("utf-8")
    assert _worker_error_from_extra(extra) == tb


@pytest.mark.parametrize(
    "extra",
    [
        b"",  # nothing
        b"\x00",  # the pre-1.0 padding byte
        b"free-form text",  # non-JSON future use
        b'{"broken json',  # unparseable
        b'{"note": "no error key"}',  # JSON object without an error
        b"[1, 2, 3]",  # JSON but not an object
        b'{"error": null}',  # explicit null error
    ],
)
def test_non_error_extras_are_ignored(extra):
    assert _worker_error_from_extra(extra) is None
