"""Application configuration.

Deployment-specific values and secrets only. Scientific parameters
(study area, polarization, thresholds) belong in configs/*.yaml so that
an analysis is reproducible from its config file and input manifest,
not from someone's shell environment.

No global Settings instance is constructed at import time. The entry
point loads settings once and passes them down.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # Container-side paths. The host directories are bind-mounted here
    # by compose.yaml; nothing is copied into the image.
    data_root: Path = Path("/data")
    config_root: Path = Path("/configs")

    processing_tile_size: int = Field(default=1024, ge=128, le=4096)
    processing_max_workers: int = Field(default=2, ge=1, le=16)

    earthdata_netrc_path: Path = Path("/run/secrets/earthdata_netrc")

    # Artifacts are published by POSTing them to the Cloudflare Worker,
    # which writes to R2 through its native binding. This container never
    # holds R2 credentials, so there is nothing here to leak.
    worker_api_url: str | None = None
    processor_shared_secret: SecretStr | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    def upload_is_configured(self) -> bool:
        """True only when artifacts can actually be published.

        Both values are required: a URL without a shared secret would
        post to an endpoint that must reject it.
        """
        return bool(self.worker_api_url and self.processor_shared_secret)


def load_settings() -> Settings:
    """Load settings from the environment. Call once, at the entry point."""
    return Settings()
