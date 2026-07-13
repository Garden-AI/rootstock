"""run_worker must tolerate options it doesn't know about.

Workers are frozen inside built envs; a newer client passing a new kwarg
must not TypeError against an already-deployed worker.
"""

from __future__ import annotations

import io

import pytest

from rootstock.worker import run_worker


def _bail_out_setup(checkpoint, device, **kwargs):
    # Short-circuit before any socket work — signature acceptance is the test.
    raise SystemExit(0)


def test_run_worker_ignores_unknown_kwargs():
    with pytest.raises(SystemExit):
        run_worker(
            setup_fn=_bail_out_setup,
            checkpoint="mace-mp-0-medium",
            device="cpu",
            socket_path="/tmp/does_not_matter",
            option_from_the_future=True,
            another_new_option={"nested": 1},
        )


def test_run_worker_logs_ignored_kwargs():
    log = io.StringIO()
    with pytest.raises(SystemExit):
        run_worker(
            setup_fn=_bail_out_setup,
            checkpoint="mace-mp-0-medium",
            device="cpu",
            socket_path="/tmp/does_not_matter",
            log=log,
            option_from_the_future=True,
        )
    assert "option_from_the_future" in log.getvalue()
