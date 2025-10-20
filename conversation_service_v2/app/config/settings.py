"""Application settings using pydantic-settings."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_env: str = "development"
    app_port: int = 3007
    app_debug: bool = True

    # JWT
    jwt_secret_key: str = "your-secret-key-ultra-securisee-256-bits"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # DeepSeek API
    deepseek_api_key: str = "sk-your-deepseek-api-key"
    deepseek_base_url: str = "https://api.deepseek.com"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "harena"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # CORS
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
