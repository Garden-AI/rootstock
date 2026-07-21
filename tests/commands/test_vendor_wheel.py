"""Vendoring the rootstock wheel into {root}/wheels/.

Rebuilds reinstall the pinned rootstock; without a vendored copy they depend
on PyPI serving that exact release, un-yanked, for the lifetime of the
install. All network access is faked; the sha256 comes from the fake PyPI
metadata just like the real API.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from rootstock.operations import _vendor_rootstock_wheel

WHEEL_BYTES = b"not-really-a-wheel-but-bytes-are-bytes"
VERSION = "9.9.9"
FILENAME = f"rootstock-{VERSION}-py3-none-any.whl"


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fake_urlopen_factory(calls, wheel_bytes=WHEEL_BYTES, advertised_sha=None):
    sha = advertised_sha or hashlib.sha256(wheel_bytes).hexdigest()
    metadata = {
        "urls": [
            {
                "packagetype": "sdist",
                "filename": f"rootstock-{VERSION}.tar.gz",
                "url": "https://files.example/sdist",
                "digests": {"sha256": "irrelevant"},
            },
            {
                "packagetype": "bdist_wheel",
                "filename": FILENAME,
                "url": "https://files.example/wheel",
                "digests": {"sha256": sha},
            },
        ]
    }

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if url.endswith("/json"):
            return _FakeResponse(json.dumps(metadata).encode())
        return _FakeResponse(wheel_bytes)

    return fake_urlopen


@pytest.fixture
def release_version(monkeypatch):
    monkeypatch.setattr("rootstock.__version__", VERSION)


def test_vendors_wheel_from_pypi(tmp_path, monkeypatch, release_version):
    calls: list[str] = []
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(calls))

    wheel = _vendor_rootstock_wheel(tmp_path)

    assert wheel == tmp_path / "wheels" / FILENAME
    assert wheel.read_bytes() == WHEEL_BYTES
    assert calls == [
        f"https://pypi.org/pypi/rootstock/{VERSION}/json",
        "https://files.example/wheel",
    ]


def test_already_vendored_wheel_is_reused_offline(tmp_path, monkeypatch, release_version):
    calls: list[str] = []
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(calls))
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / FILENAME).write_bytes(WHEEL_BYTES)

    wheel = _vendor_rootstock_wheel(tmp_path)

    assert wheel == wheels / FILENAME
    assert calls == []  # a rebuild must not need the network for this


def test_dev_builds_are_not_vendored(tmp_path, monkeypatch):
    monkeypatch.setattr("rootstock.__version__", "9.9.9.dev0+abc1234")
    calls: list[str] = []
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(calls))

    assert _vendor_rootstock_wheel(tmp_path) is None
    assert calls == []


def test_sha_mismatch_rejects_wheel(tmp_path, monkeypatch, release_version, capsys):
    calls: list[str] = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(calls, advertised_sha="0" * 64),
    )

    assert _vendor_rootstock_wheel(tmp_path) is None
    assert "sha256 mismatch" in capsys.readouterr().err
    assert not list((tmp_path / "wheels").glob("*.whl")) if (tmp_path / "wheels").exists() else True


def test_network_failure_falls_back_with_warning(tmp_path, monkeypatch, release_version, capsys):
    def failing_urlopen(url, timeout=None):
        raise urllib.error.URLError("no route to pypi")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)

    assert _vendor_rootstock_wheel(tmp_path) is None
    assert "could not vendor" in capsys.readouterr().err


def test_install_step_prefers_vendored_wheel(tmp_path, monkeypatch, release_version, capsys):
    """The env's rootstock comes from the vendored file, not the index."""
    from unittest.mock import MagicMock

    calls: list[str] = []
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(calls))

    env_dir = tmp_path / "environments"
    env_dir.mkdir()
    env_source = env_dir / "noop.py"
    env_source.write_text(
        "# /// script\n"
        '# requires-python = ">=3.10"\n'
        "# dependencies = []\n"
        "# ///\n"
        "CHECKPOINTS = {}\n"
        'def setup(checkpoint: str, device: str = "cuda"):\n'
        "    return None\n"
    )

    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        if cmd[:2] == ["uv", "venv"]:
            Path(cmd[2]).mkdir(parents=True, exist_ok=True)
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = ""
        return result

    monkeypatch.setattr("rootstock.operations.subprocess.run", fake_run)
    monkeypatch.setattr("rootstock.operations.shutil.copy", lambda *a, **k: None)
    monkeypatch.setattr("rootstock.operations._precompile_environment", lambda *a, **k: None)
    monkeypatch.setattr("rootstock.operations.update_and_push_manifest", lambda *a, **k: None)

    from rootstock.operations import install_environment

    install_environment(root=tmp_path, source=str(env_source), force=False, verbose=False)

    wheel_path = str(tmp_path / "wheels" / FILENAME)
    pip_installs = [c for c in captured if c[:3] == ["uv", "pip", "install"]]
    assert pip_installs and pip_installs[0][-1] == wheel_path
