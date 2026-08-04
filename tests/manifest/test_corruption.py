"""A corrupted manifest fails loudly instead of reading as missing.

``load_manifest`` used to swallow parse errors and return None; callers then
created a fresh manifest and the next save overwrote all fetch/verify
history without a word. Now the file's absence is the only thing that means
"no manifest" — anything else raises ManifestError, which the CLI turns
into a clean error message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rootstock.config import UserConfig
from rootstock.manifest import (
    ManifestError,
    create_manifest,
    load_manifest,
    save_manifest,
)


def test_missing_manifest_is_none(tmp_path: Path):
    assert load_manifest(tmp_path) is None


def test_corrupt_json_raises_with_path(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{not json")

    with pytest.raises(ManifestError, match=str(tmp_path / "manifest.json")):
        load_manifest(tmp_path)


def test_missing_required_field_raises(tmp_path: Path):
    """Valid JSON with required keys stripped is corruption, not absence."""
    manifest = create_manifest(tmp_path, ["test"], UserConfig(name="t", email="t@t.t"))
    save_manifest(manifest, tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    del data["maintainer"]
    (tmp_path / "manifest.json").write_text(json.dumps(data))

    with pytest.raises(ManifestError, match="refusing to treat it as missing"):
        load_manifest(tmp_path)


def test_wrong_top_level_type_raises(tmp_path: Path):
    (tmp_path / "manifest.json").write_text('["not", "a", "manifest"]')

    with pytest.raises(ManifestError):
        load_manifest(tmp_path)


def test_valid_manifest_still_loads(tmp_path: Path):
    manifest = create_manifest(tmp_path, ["test"], UserConfig(name="t", email="t@t.t"))
    save_manifest(manifest, tmp_path)

    loaded = load_manifest(tmp_path)

    assert loaded is not None
    assert loaded.clusters == ["test"]


def test_newer_schema_raises_manifest_error(tmp_path: Path):
    """Schema mismatches share the taxonomy: a clean ManifestError, not a
    bare RuntimeError traceback."""
    manifest = create_manifest(tmp_path, ["test"], UserConfig(name="t", email="t@t.t"))
    save_manifest(manifest, tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["schema_version"] = 99
    (tmp_path / "manifest.json").write_text(json.dumps(data))

    with pytest.raises(ManifestError, match="upgrade this client"):
        load_manifest(tmp_path)


def test_cli_reports_corruption_cleanly(tmp_path: Path, monkeypatch, capsys):
    """End to end: `rootstock status` on a corrupt manifest exits 1 with a
    clean error, no traceback, and does not clobber the file."""
    import rootstock.cli as cli

    (tmp_path / "manifest.json").write_text("{not json")
    before = (tmp_path / "manifest.json").read_text()

    monkeypatch.setattr("sys.argv", ["rootstock", "status", "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "corrupted" in err
    assert "Traceback" not in err
    assert (tmp_path / "manifest.json").read_text() == before
