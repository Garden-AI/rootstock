"""Tests for _rootstock_install_spec — branches on dev vs clean __version__.

These tests exercise the helper as pure string manipulation. The input is
``rootstock.__version__`` (controlled via ``monkeypatch.setattr``), the
output is the install spec string. ``importlib.metadata`` is NOT mocked —
the helper reads the live attribute, which is itself set from metadata at
package load time.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rootstock.commands.install import (
    ROOTSTOCK_GITHUB_URL,
    _rootstock_install_spec,
)


def test_clean_tagged_release_returns_pypi_spec(monkeypatch):
    monkeypatch.setattr("rootstock.__version__", "0.7.2")

    assert _rootstock_install_spec() == "rootstock==0.7.2"


def test_post_release_without_dev_treated_as_clean(monkeypatch):
    monkeypatch.setattr("rootstock.__version__", "0.7.2.post1")

    assert _rootstock_install_spec() == "rootstock==0.7.2.post1"


def test_dev_version_with_short_sha_returns_git_url(monkeypatch):
    monkeypatch.setattr("rootstock.__version__", "0.7.1.post3.dev0+abc1234")

    assert _rootstock_install_spec() == (f"rootstock@git+{ROOTSTOCK_GITHUB_URL}@abc1234")


def test_dev_version_with_long_sha_preserved_verbatim(monkeypatch):
    monkeypatch.setattr("rootstock.__version__", "0.7.1.post1.dev0+abcdef0123456")

    spec = _rootstock_install_spec()

    assert spec.endswith("@abcdef0123456")


def test_dev_version_with_dirty_marker_passes_local_segment_through(monkeypatch):
    """uv-dynamic-versioning may emit '<sha>.dirty' as the local segment.

    The helper splits on the final '+' and uses everything after it as the
    git ref. We document that behavior here: the '.dirty' suffix is passed
    through verbatim (uv pip install will reject it at install time, which
    is acceptable — a dirty tree shouldn't be producing reproducible envs).
    """
    monkeypatch.setattr("rootstock.__version__", "0.7.1.post1.dev0+abc1234.dirty")

    spec = _rootstock_install_spec()

    assert spec == f"rootstock@git+{ROOTSTOCK_GITHUB_URL}@abc1234.dirty"


def test_dev_version_missing_plus_sha_raises_runtime_error(monkeypatch):
    monkeypatch.setattr("rootstock.__version__", "0.0.0.dev0")

    with pytest.raises(RuntimeError, match="Cannot determine rootstock commit"):
        _rootstock_install_spec()


def test_dev_version_missing_plus_sha_error_includes_offending_value(monkeypatch):
    monkeypatch.setattr("rootstock.__version__", "0.0.0.dev0")

    with pytest.raises(RuntimeError, match="0.0.0.dev0"):
        _rootstock_install_spec()


def test_install_command_passes_helper_spec_as_final_arg(tmp_path, monkeypatch, capsys):
    """The install command surfaces _rootstock_install_spec()'s output.

    This is an integration test verifying the wiring between the helper and
    the call to ``uv pip install``. ``subprocess.run`` is mocked at the
    module boundary; the helper itself is NOT mocked (its real output is
    what we want to flow through).
    """
    monkeypatch.setattr("rootstock.__version__", "9.9.9")

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

    captured_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured_calls.append(list(cmd))
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = ""
        return result

    monkeypatch.setattr("rootstock.commands.install.subprocess.run", fake_run)
    monkeypatch.setattr("rootstock.commands.install.shutil.copy", lambda *a, **k: None)
    monkeypatch.setattr(
        "rootstock.commands.manifest.update_and_push_manifest",
        lambda *a, **k: None,
    )

    from rootstock.commands.install import _install_single_environment

    rc = _install_single_environment(
        root=tmp_path,
        source=str(env_source),
        force=False,
        verbose=False,
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert "Installing: rootstock==9.9.9" in captured.out

    rootstock_install_calls = [
        c
        for c in captured_calls
        if len(c) >= 3 and c[0:3] == ["uv", "pip", "install"] and c[-1] == "rootstock==9.9.9"
    ]
    assert len(rootstock_install_calls) == 1, (
        f"expected exactly one 'uv pip install ... rootstock==9.9.9' call, "
        f"got calls: {captured_calls!r}"
    )


def test_dependencies_installed_via_uv_sync_script(tmp_path, monkeypatch, capsys):
    """Env dependencies install through `uv sync --script`, not `uv pip install`.

    `uv pip install` ignores `[tool.uv.sources]`/`[[tool.uv.index]]`, so a torch
    pin to a CUDA-specific index would be silently dropped. The deps step must
    go through the script interface (`uv sync --script ... --active`) with
    VIRTUAL_ENV pointed at the freshly-built venv so the env source's full uv
    config is honored.
    """
    monkeypatch.setattr("rootstock.__version__", "9.9.9")

    env_dir = tmp_path / "environments"
    env_dir.mkdir()
    env_source = env_dir / "withdeps.py"
    env_source.write_text(
        "# /// script\n"
        '# requires-python = ">=3.10"\n'
        '# dependencies = ["torch>=2.0"]\n'
        "#\n"
        "# [tool.uv.sources]\n"
        '# torch = { index = "pytorch-cu128" }\n'
        "#\n"
        "# [[tool.uv.index]]\n"
        '# name = "pytorch-cu128"\n'
        '# url = "https://download.pytorch.org/whl/cu128"\n'
        "# explicit = true\n"
        "# ///\n"
        "CHECKPOINTS = {}\n"
        'def setup(checkpoint: str, device: str = "cuda"):\n'
        "    return None\n"
    )

    captured_calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        captured_calls.append((list(cmd), kwargs))
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = ""
        return result

    monkeypatch.setattr("rootstock.commands.install.subprocess.run", fake_run)
    monkeypatch.setattr("rootstock.commands.install.shutil.copy", lambda *a, **k: None)
    monkeypatch.setattr(
        "rootstock.commands.manifest.update_and_push_manifest",
        lambda *a, **k: None,
    )

    from rootstock.commands.install import _install_single_environment

    rc = _install_single_environment(
        root=tmp_path,
        source=str(env_source),
        force=False,
        verbose=False,
    )
    assert rc == 0

    env_target = str(tmp_path / "envs" / "withdeps")

    sync_calls = [
        (cmd, kwargs)
        for cmd, kwargs in captured_calls
        if cmd[:2] == ["uv", "sync"]
    ]
    assert len(sync_calls) == 1, (
        f"expected exactly one 'uv sync' call, got: {[c for c, _ in captured_calls]!r}"
    )
    cmd, kwargs = sync_calls[0]
    assert cmd == ["uv", "sync", "--script", str(env_source), "--active"]
    assert kwargs["env"]["VIRTUAL_ENV"] == env_target

    # The dependency must NOT be installed through the `uv pip` interface,
    # which would ignore the pinned CUDA index.
    pip_dep_calls = [
        cmd
        for cmd, _ in captured_calls
        if cmd[:3] == ["uv", "pip", "install"] and any("torch" in arg for arg in cmd)
    ]
    assert pip_dep_calls == [], (
        f"dependencies must not be installed via 'uv pip install': {pip_dep_calls!r}"
    )
