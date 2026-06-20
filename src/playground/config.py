"""Application settings loaded from environment variables."""

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
