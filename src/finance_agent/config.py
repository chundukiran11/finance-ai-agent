"""Application configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["gemini", "openai", "anthropic"] = "gemini"
    llm_model: str = "gemini-2.0-flash"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    database_path: str = "./data/finance.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8001


@lru_cache
def get_settings() -> Settings:
    return Settings()
