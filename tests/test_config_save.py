"""save_config writes real TOML: awkward values must round-trip.

The old writer built TOML by string concatenation with no escaping while
reading with ``tomllib`` — a value containing a double quote produced a
config file that could never be read back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootstock.config import UserConfig, load_config, save_config


def _round_trip(tmp_path: Path, config: UserConfig) -> UserConfig:
    path = tmp_path / "config.toml"
    save_config(config, config_path=path)
    return load_config(config_path=path)


def test_plain_config_round_trips(tmp_path: Path):
    config = UserConfig(
        root="/scratch/rootstock",
        name="Owen",
        email="owen@example.edu",
        is_maintainer=True,
    )

    loaded = _round_trip(tmp_path, config)

    assert loaded.root == "/scratch/rootstock"
    assert loaded.name == "Owen"
    assert loaded.email == "owen@example.edu"
    assert loaded.is_maintainer is True


@pytest.mark.parametrize(
    "hostile",
    [
        'quo"te',
        "back\\slash",
        "new\nline",
        "uniçode — dash",
        "'single' and \"double\"",
    ],
)
def test_hostile_values_round_trip(tmp_path: Path, hostile: str):
    """The exact failure mode of the string-concat writer: these produced
    unparseable files or silently corrupted values."""
    config = UserConfig(name=hostile, api_key=hostile, root=hostile)

    loaded = _round_trip(tmp_path, config)

    assert loaded.name == hostile
    assert loaded.api_key == hostile
    assert loaded.root == hostile


def test_unset_fields_are_omitted(tmp_path: Path):
    path = tmp_path / "config.toml"
    save_config(UserConfig(), config_path=path)

    text = path.read_text()
    assert "api_key" not in text
    assert "[maintainer]" not in text
    assert "is_maintainer = false" in text


def test_usage_api_url_round_trips(tmp_path: Path):
    config = UserConfig(usage_api_url="https://example.com/usage")
    assert _round_trip(tmp_path, config).usage_api_url == "https://example.com/usage"


def test_usage_url_derived_from_standard_api_url():
    config = UserConfig(api_url="https://garden-ai-prod--rootstock-admin-manifest.modal.run")
    assert (
        config.resolve_usage_api_url() == "https://garden-ai-prod--rootstock-admin-usage.modal.run"
    )


def test_explicit_usage_url_wins_over_derivation():
    config = UserConfig(
        api_url="https://garden-ai-prod--rootstock-admin-manifest.modal.run",
        usage_api_url="https://example.com/usage",
    )
    assert config.resolve_usage_api_url() == "https://example.com/usage"


def test_nonstandard_api_url_does_not_derive():
    assert UserConfig(api_url="https://example.com/ingest").resolve_usage_api_url() is None
    assert UserConfig().resolve_usage_api_url() is None
