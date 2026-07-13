"""Worker-side configuration, read once from ROOTSTOCK_* env vars at startup.

This module executes inside built environments alongside worker.py and
protocol.py, so it is part of the frozen worker surface: every built env
pins its own copy, and environment variables set at spawn are the one
configuration channel a newer client (or an operator) can always reach an
already-deployed worker through.

Rules for this module, which exist because it can never be hotfixed:

- Keep it dependency-free (stdlib only) and free of imports from the
  client-side half of rootstock.
- Every knob has a safe default; an unset, empty, or invalid value must
  never raise. Problems are collected as warnings on the config object and
  logged once at worker startup instead.
- New knobs may be added in later releases (old workers simply won't read
  them); existing knob names and semantics are frozen.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache

CONNECT_RETRIES_ENV = "ROOTSTOCK_WORKER_CONNECT_RETRIES"
CONNECT_RETRY_DELAY_ENV = "ROOTSTOCK_WORKER_CONNECT_RETRY_DELAY"
WORKER_LOG_ENV = "ROOTSTOCK_WORKER_LOG"

DEFAULT_CONNECT_RETRIES = 50
DEFAULT_CONNECT_RETRY_DELAY = 0.1


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable snapshot of the worker's env-var knobs.

    Attributes:
        connect_retries: Connection attempts to the server socket.
        connect_retry_delay: Seconds between connection attempts.
        log_target: Raw ROOTSTOCK_WORKER_LOG value ("stderr", "stdout",
            or a file path); resolve with open_log().
        warnings: Human-readable notices about ignored invalid values,
            for the worker to log once at startup.
    """

    connect_retries: int = DEFAULT_CONNECT_RETRIES
    connect_retry_delay: float = DEFAULT_CONNECT_RETRY_DELAY
    log_target: str | None = None
    warnings: tuple[str, ...] = field(default=())

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkerConfig:
        """Build a config from environ (default: os.environ). Never raises."""
        env = os.environ if environ is None else environ
        warnings: list[str] = []

        def number(name: str, default, cast):
            raw = env.get(name)
            if raw is None:
                return default
            try:
                return cast(raw)
            except ValueError:
                warnings.append(
                    f"Ignoring invalid {name}={raw!r} (expected {cast.__name__}), using {default}"
                )
                return default

        return cls(
            connect_retries=number(CONNECT_RETRIES_ENV, DEFAULT_CONNECT_RETRIES, int),
            connect_retry_delay=number(CONNECT_RETRY_DELAY_ENV, DEFAULT_CONNECT_RETRY_DELAY, float),
            log_target=env.get(WORKER_LOG_ENV) or None,
            warnings=tuple(warnings),
        )

    def open_log(self):
        """Resolve log_target into a writable file object (or None).

        "stderr" and "stdout" map to the process streams; anything else is
        opened as a file path in append mode, line-buffered. An unopenable
        path is ignored — a bad log destination must never kill a worker.
        """
        if not self.log_target:
            return None
        if self.log_target == "stderr":
            return sys.stderr
        if self.log_target == "stdout":
            return sys.stdout
        try:
            return open(self.log_target, "a", buffering=1)
        except OSError:
            return None


@lru_cache(maxsize=1)
def get_worker_config() -> WorkerConfig:
    """The process-wide config singleton, read from os.environ on first use.

    A worker process is spawned fresh for every session, so reading once is
    what "read at startup" means. Tests can reset with
    get_worker_config.cache_clear().
    """
    return WorkerConfig.from_env()
