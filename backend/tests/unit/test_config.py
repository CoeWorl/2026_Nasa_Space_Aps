"""Configuration contract tests.

These run without Docker, without data and without credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from groundtruth.config import Settings


def _settings(**overrides) -> Settings:
    # _env_file=None keeps a developer's local .env out of the test run.
    return Settings(_env_file=None, **overrides)


def test_defaults_point_at_the_container_mounts() -> None:
    settings = _settings()
    assert settings.data_root == Path("/data")
    assert settings.config_root == Path("/configs")


def test_derived_data_directories() -> None:
    settings = _settings(data_root=Path("/tmp/gt"))
    assert settings.raw_dir == Path("/tmp/gt/raw")
    assert settings.interim_dir == Path("/tmp/gt/interim")
    assert settings.processed_dir == Path("/tmp/gt/processed")


def test_upload_is_not_configured_by_default() -> None:
    assert _settings().upload_is_configured() is False


def test_upload_needs_both_the_url_and_the_shared_secret() -> None:
    url_only = _settings(worker_api_url="https://api.example")
    assert url_only.upload_is_configured() is False

    secret_only = _settings(processor_shared_secret="s3cret")
    assert secret_only.upload_is_configured() is False

    complete = _settings(
        worker_api_url="https://api.example",
        processor_shared_secret="s3cret",
    )
    assert complete.upload_is_configured() is True


def test_secrets_are_not_printed_in_the_repr() -> None:
    settings = _settings(processor_shared_secret="super-secret-value")
    assert "super-secret-value" not in repr(settings)


@pytest.mark.parametrize("size", [64, 8192])
def test_tile_size_is_bounded(size: int) -> None:
    with pytest.raises(ValueError):
        _settings(processing_tile_size=size)


def test_environment_variables_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/mnt/elsewhere")
    monkeypatch.setenv("PROCESSING_TILE_SIZE", "512")
    settings = Settings(_env_file=None)
    assert settings.data_root == Path("/mnt/elsewhere")
    assert settings.processing_tile_size == 512
