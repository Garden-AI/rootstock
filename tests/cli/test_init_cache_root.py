"""Tests for how ``rootstock init`` settles the model-weight cache root."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rootstock.commands.init import resolve_init_cache_root


def _args(**overrides):
    base = dict(cache_root=None, skip_dirs=False, skip_manifest=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def _answers(monkeypatch, *responses):
    """Feed successive replies to the prompts, failing on an unexpected one.

    Returns the list the prompts land in — monkeypatching ``input`` means the
    prompt text never reaches stdout, so assertions about defaults read here.
    """
    queue = list(responses)
    prompts: list[str] = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"unexpected extra prompt: {prompt!r}")
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def test_explicit_flag_wins_without_prompting(monkeypatch):
    _answers(monkeypatch)  # any prompt at all is a failure
    got = resolve_init_cache_root(
        _args(cache_root="/scratch/me/rs-cache"), Path("/install"), "perlmutter"
    )
    assert got == Path("/scratch/me/rs-cache")


def test_explicit_flag_expands_user(monkeypatch):
    _answers(monkeypatch)
    got = resolve_init_cache_root(_args(cache_root="~/rs-cache"), Path("/install"), None)
    assert got == Path.home() / "rs-cache"


def test_explicit_flag_keeps_symlinks_uncollapsed(monkeypatch, tmp_path):
    """The declaration is read by other users' clients, so keep it verbatim."""
    real = tmp_path / "real-cache"
    real.mkdir()
    link = tmp_path / "link-cache"
    link.symlink_to(real)

    _answers(monkeypatch)
    got = resolve_init_cache_root(_args(cache_root=str(link)), Path("/install"), None)
    assert got == link


def test_declining_the_split_uses_install_root(monkeypatch):
    _answers(monkeypatch, "n")
    assert resolve_init_cache_root(_args(), Path("/install"), None) == Path("/install")


def test_accepting_the_split_takes_the_typed_path(monkeypatch):
    _answers(monkeypatch, "y", "/scratch/me/rs-cache")
    got = resolve_init_cache_root(_args(), Path("/install"), None)
    assert got == Path("/scratch/me/rs-cache")


def test_unknown_cluster_defaults_to_no_split(monkeypatch):
    """With no registry hint the default answer is 'same filesystem'."""
    prompts = _answers(monkeypatch, "")  # accept the default
    assert resolve_init_cache_root(_args(), Path("/install"), None) == Path("/install")
    assert "(y/n) [n]" in prompts[0]


def test_registry_split_is_offered_as_the_default(monkeypatch):
    """Perlmutter is registered split, so both prompts should pre-fill."""
    from rootstock.clusters import get_cluster

    expected = get_cluster("perlmutter").resolved_cache_root
    prompts = _answers(monkeypatch, "", "")  # accept both defaults
    got = resolve_init_cache_root(_args(), get_cluster("perlmutter").root, "perlmutter")
    assert got == expected

    assert "(y/n) [y]" in prompts[0]
    assert str(expected) in prompts[1]


def test_registry_suggestion_is_overridable(monkeypatch):
    """The whole point: a stale baked-in registry must not bind the install."""
    from rootstock.clusters import get_cluster

    _answers(monkeypatch, "y", "/pscratch/somewhere/else")
    got = resolve_init_cache_root(_args(), get_cluster("perlmutter").root, "perlmutter")
    assert got == Path("/pscratch/somewhere/else")


def test_registry_split_can_be_declined(monkeypatch):
    from rootstock.clusters import get_cluster

    root = get_cluster("perlmutter").root
    _answers(monkeypatch, "n")
    assert resolve_init_cache_root(_args(), root, "perlmutter") == root


def test_empty_answer_without_a_default_reprompts(monkeypatch, capsys):
    _answers(monkeypatch, "y", "", "/scratch/me/rs-cache")
    got = resolve_init_cache_root(_args(), Path("/install"), None)
    assert got == Path("/scratch/me/rs-cache")
    assert "A cache root is required" in capsys.readouterr().err


def test_skip_dirs_asks_nothing(monkeypatch, tmp_path):
    """No layout marker gets written, so there is nothing to ask about."""
    from rootstock.commands.init import cmd_init

    _answers(monkeypatch, str(tmp_path), "n")  # root, then maintainer? -> n
    monkeypatch.setattr("rootstock.commands.init.save_config", lambda config: None)
    rc = cmd_init(_args(skip_dirs=True))
    assert rc == 0
    assert not (tmp_path / "layout.json").exists()


@pytest.mark.parametrize("reply", ["y", "Y", "yes", "YES"])
def test_yes_spellings(monkeypatch, reply):
    _answers(monkeypatch, reply, "/scratch/c")
    assert resolve_init_cache_root(_args(), Path("/install"), None) == Path("/scratch/c")
