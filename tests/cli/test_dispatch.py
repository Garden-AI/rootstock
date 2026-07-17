"""Tests for main()'s argument dispatch: benchmark forwarding and strictness."""

from __future__ import annotations

import pytest

from rootstock import cli


def _run_main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["rootstock", *argv])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return excinfo.value.code


def test_benchmark_forwards_leading_options(monkeypatch):
    """argparse.REMAINDER chokes on a leading option like --list; the
    parse_known_args forwarding must pass it through untouched."""
    seen = {}

    def fake_benchmark(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "cmd_benchmark", fake_benchmark)
    assert _run_main(monkeypatch, ["benchmark", "--list"]) == 0
    assert seen["argv"] == ["--list"]


def test_benchmark_forwards_multiple_args_in_order(monkeypatch):
    seen = {}

    def fake_benchmark(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "cmd_benchmark", fake_benchmark)
    _run_main(monkeypatch, ["benchmark", "--devices", "cuda", "cpu", "--calls", "5"])
    assert seen["argv"] == ["--devices", "cuda", "cpu", "--calls", "5"]


def test_non_benchmark_commands_reject_unknown_args(monkeypatch):
    """Forwarding leftovers to benchmark must not loosen any other command."""
    assert _run_main(monkeypatch, ["list", "--bogus"]) == 2


def test_manifest_subcommands_dispatch_directly(monkeypatch):
    """Each manifest subparser binds its own func — no re-dispatch layer."""
    called = []
    monkeypatch.setattr(cli, "cmd_manifest_show", lambda args: called.append("show") or 0)
    assert _run_main(monkeypatch, ["manifest", "show", "--root", "/nowhere"]) == 0
    assert called == ["show"]
