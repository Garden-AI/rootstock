"""Anonymous usage records: spooled per session, and never able to fail loudly.

The spool contract (issue #47): one JSON file per worker session under
``{cache_root}/usage/``, written only when a maintainer provisioned that
directory, skippable by env var, and swallowing every possible error —
telemetry must never break or block a calculation.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from rootstock.server import RootstockServer
from rootstock.usage import (
    DISABLE_ENV_VAR,
    LOCAL_CHECKPOINT_LABEL,
    RECORD_SCHEMA_VERSION,
    record_session,
    usage_dir,
    usage_disabled,
)


def _write(cache_root, **overrides):
    kwargs = dict(
        root=cache_root,
        cache_root=cache_root,
        env_name="mace",
        checkpoint="mace-mp-0-medium",
        is_local=False,
        device="cuda",
        client="calculator",
        started_at="2026-07-23T01:02:03+00:00",
        duration_s=12.34,
        n_calculations=7,
    )
    kwargs.update(overrides)
    return record_session(**kwargs)


def test_record_written_when_spool_provisioned(tmp_path):
    usage_dir(tmp_path).mkdir()
    path = _write(tmp_path)

    assert path is not None
    assert path.parent == usage_dir(tmp_path)
    record = json.loads(path.read_text())
    assert record == {
        "schema_version": RECORD_SCHEMA_VERSION,
        "started_at": "2026-07-23T01:02:03+00:00",
        "duration_s": 12.3,
        "cluster": None,  # tmp_path is not a registered cluster root
        "root": str(tmp_path),
        "env": "mace",
        "checkpoint": "mace-mp-0-medium",
        "device": "cuda",
        "client": "calculator",
        "rootstock_version": record["rootstock_version"],
        "n_calculations": 7,
        "user": record["user"],
    }


def test_serve_record_carries_client_and_null_call_count(tmp_path):
    """serve's parent process can't see the i-PI traffic, so its records
    carry n_calculations=null — the client field still says who it was."""
    usage_dir(tmp_path).mkdir()
    path = _write(tmp_path, client="serve", n_calculations=None)
    record = json.loads(path.read_text())
    assert record["client"] == "serve"
    assert record["n_calculations"] is None


def test_unprovisioned_spool_is_a_silent_noop(tmp_path):
    """No usage/ dir means the install opted out — and record_session must
    not create the dir itself (a user-created spool would be owned by
    whoever ran first and unwritable to everyone else)."""
    assert _write(tmp_path) is None
    assert not usage_dir(tmp_path).exists()


def test_records_are_unique_per_session(tmp_path):
    usage_dir(tmp_path).mkdir()
    paths = {_write(tmp_path) for _ in range(3)}
    assert len(paths) == 3


def test_local_checkpoint_id_is_masked(tmp_path):
    """Ids registered via add-local are user-chosen names; anonymous stats
    must not leak them."""
    usage_dir(tmp_path).mkdir()
    path = _write(tmp_path, checkpoint="my-secret-project-ft", is_local=True)
    assert json.loads(path.read_text())["checkpoint"] == LOCAL_CHECKPOINT_LABEL


def test_custom_checkpoint_id_recorded_verbatim(tmp_path):
    """<family>:custom is exempt from the mask — the marker already
    self-flags the run as user weights and the id is env-declared, not
    user-chosen. Exempt even when the entry point reports is_local (the
    server flags any session with a checkpoint_path)."""
    usage_dir(tmp_path).mkdir()
    path = _write(tmp_path, checkpoint="uma:custom", is_local=True)
    assert json.loads(path.read_text())["checkpoint"] == "uma:custom"


def test_env_var_opt_out(tmp_path, monkeypatch):
    usage_dir(tmp_path).mkdir()
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    assert _write(tmp_path) is None
    assert list(usage_dir(tmp_path).iterdir()) == []


@pytest.mark.parametrize(
    ("value", "disabled"),
    [("", False), ("0", False), ("false", False), ("1", True), ("true", True), ("TRUE", True)],
)
def test_usage_disabled_parsing(monkeypatch, value, disabled):
    monkeypatch.setenv(DISABLE_ENV_VAR, value)
    assert usage_disabled() is disabled


# --------------------------------------------------------------------------- #
# User hash (distinct-user counting)
# --------------------------------------------------------------------------- #


def _user_of(path):
    return json.loads(path.read_text())["user"]


def test_user_hash_is_salted_stable_and_truncated(tmp_path):
    usage_dir(tmp_path).mkdir()
    first = _user_of(_write(tmp_path))
    second = _user_of(_write(tmp_path))

    assert first == second  # stable per user per install: distinct counts work
    assert len(first) == 16
    int(first, 16)  # hex, i.e. a hash — not a raw username

    import getpass
    import hashlib

    # Salted: the raw and unsalted-hashed username must not appear anywhere.
    username = getpass.getuser()
    assert first != username
    assert first != hashlib.sha256(username.encode()).hexdigest()[:16]


def test_user_hash_differs_per_user_and_per_install(tmp_path, monkeypatch):
    install_a = tmp_path / "a"
    install_b = tmp_path / "b"
    for install in (install_a, install_b):
        install.mkdir()
        usage_dir(install).mkdir()

    alice_a = _user_of(_write(install_a))
    monkeypatch.setattr("rootstock.usage.getpass.getuser", lambda: "somebody-else")
    bob_a = _user_of(_write(install_a))
    bob_b = _user_of(_write(install_b))

    assert alice_a != bob_a  # different users, same install
    assert bob_a != bob_b  # same user, different installs (different salts)


def test_salt_created_once_world_readable(tmp_path):
    usage_dir(tmp_path).mkdir()
    _write(tmp_path)
    salt_path = usage_dir(tmp_path) / "salt"
    assert salt_path.is_file()
    assert stat.S_IMODE(salt_path.stat().st_mode) == 0o444
    salt = salt_path.read_bytes()
    assert len(salt) == 32

    _write(tmp_path)
    assert salt_path.read_bytes() == salt  # first writer won; never rewritten


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores file modes")
def test_unreadable_salt_drops_user_field_not_the_record(tmp_path):
    usage_dir(tmp_path).mkdir()
    salt_path = usage_dir(tmp_path) / "salt"
    salt_path.write_bytes(b"x" * 32)
    salt_path.chmod(0o000)
    try:
        path = _write(tmp_path)
    finally:
        salt_path.chmod(0o444)

    assert path is not None  # the session is still counted
    assert json.loads(path.read_text())["user"] is None


@pytest.mark.skipif(os.getuid() == 0, reason="root ignores directory modes")
def test_unwritable_spool_never_raises(tmp_path):
    spool = usage_dir(tmp_path)
    spool.mkdir()
    spool.chmod(0o555)
    try:
        assert _write(tmp_path) is None
    finally:
        spool.chmod(0o755)


# --------------------------------------------------------------------------- #
# Server hook
# --------------------------------------------------------------------------- #


def _server(tmp_path, **kwargs):
    return RootstockServer(
        env_name="mace",
        checkpoint="mace-mp-0-medium",
        device="cpu",
        root=tmp_path,
        **kwargs,
    )


def _fake_session(server):
    """Give a never-started server the state _accept_connection would set."""
    server._session_started_at = "2026-07-23T01:02:03+00:00"
    server._session_started_monotonic = 0.0
    server._n_calculations = 5


def test_stop_records_one_session(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("rootstock.server.record_session", lambda **kw: calls.append(kw))

    server = _server(tmp_path)
    _fake_session(server)
    server.stop()
    server.stop()  # idempotent teardown must not double-count

    assert len(calls) == 1
    kw = calls[0]
    assert kw["root"] == tmp_path
    assert kw["env_name"] == "mace"
    assert kw["checkpoint"] == "mace-mp-0-medium"
    assert kw["is_local"] is False
    assert kw["device"] == "cpu"
    assert kw["client"] == "calculator"
    assert kw["n_calculations"] == 5
    assert kw["started_at"] == "2026-07-23T01:02:03+00:00"


def test_stop_marks_local_checkpoint_sessions(tmp_path, monkeypatch):
    """A server running a user-registered weights file reports is_local, so
    record_session masks the user-chosen id."""
    calls = []
    monkeypatch.setattr("rootstock.server.record_session", lambda **kw: calls.append(kw))

    server = _server(tmp_path, checkpoint_path="/home/someone/weights.pt")
    _fake_session(server)
    server.stop()

    assert calls[0]["is_local"] is True


def test_stop_with_usage_client_none_records_nothing(tmp_path, monkeypatch):
    """usage_client=None opts a server out of recording entirely — verify.py
    passes it so nightly smoke-test sessions never pollute the spool."""
    calls = []
    monkeypatch.setattr("rootstock.server.record_session", lambda **kw: calls.append(kw))

    server = _server(tmp_path, usage_client=None)
    _fake_session(server)
    server.stop()

    assert calls == []


def test_stop_without_session_records_nothing(tmp_path, monkeypatch):
    """A server whose worker never connected (or that never started) was not
    a usage session."""
    calls = []
    monkeypatch.setattr("rootstock.server.record_session", lambda **kw: calls.append(kw))

    _server(tmp_path).stop()
    assert calls == []


def test_stop_resolves_cache_root_like_every_other_entry_point(tmp_path, monkeypatch):
    """cache_root=None goes through resolve_cache_root, so the spool location
    can't disagree with where the CLI thinks the cache half lives."""
    calls = []
    monkeypatch.setattr("rootstock.server.record_session", lambda **kw: calls.append(kw))

    server = _server(tmp_path, cache_root=None)
    _fake_session(server)
    server.stop()
    assert calls[0]["cache_root"] == tmp_path  # no declaration -> the root itself


def test_stop_writes_a_real_record_end_to_end(tmp_path):
    usage_dir(tmp_path).mkdir()
    server = _server(tmp_path, cache_root=tmp_path)
    _fake_session(server)
    server.stop()

    (path,) = usage_dir(tmp_path).glob("*.json")
    record = json.loads(path.read_text())
    assert record["env"] == "mace"
    assert record["n_calculations"] == 5
    assert record["duration_s"] >= 0
