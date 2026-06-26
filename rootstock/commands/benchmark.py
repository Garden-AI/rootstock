"""Benchmark command: i-PI IPC overhead vs. in-env direct calls.

Thin wrapper that forwards all arguments to ``rootstock.benchmark.main`` so the
benchmark's own argument parser stays the single source of truth. See
``rootstock benchmark --help`` for the full option list.
"""

from __future__ import annotations


def cmd_benchmark(argv) -> int:
    from ..benchmark import main

    return main(argv)
