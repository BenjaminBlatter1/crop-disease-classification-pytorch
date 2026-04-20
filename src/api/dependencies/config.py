"""
Configuration management for the inference API.

This module defines the Settings class, which loads configuration values
from environment variables or a `.env` file. These settings control model
paths, device selection, logging behavior, and API metadata.

The `get_settings()` function returns a cached Settings instance to avoid
re-parsing environment variables on every request.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    api_name: str = "Crop Disease Inference API"
    api_version: str = "v1"

    model_path: str = "weights/model_checkpoint.pth"
    device: str = "auto"  # "cpu", "cuda", or "auto"
    log_level: str = "INFO"

    max_image_size_mb: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
