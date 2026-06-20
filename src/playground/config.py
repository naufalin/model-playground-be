"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://localhost:5432/playground"

    # Auth
    secret_key: str  # required, from .env
    algorithm: str = "HS256"
    access_token_expire_hours: int = 24

    # Agent Runtime
    agent_runtime_url: str = "http://localhost:8000"

    # CORS
    cors_origins: list[str] = ["*"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
