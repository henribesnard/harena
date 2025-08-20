from pydantic_settings import BaseSettings, SettingsConfigDict


class AutoGenSettings(BaseSettings):
    """Paramètres généraux et spécifiques à AutoGen."""

    ENVIRONMENT: str = "development"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"
    PORT: int = 8000
    DEBUG: bool = False
    SECRET_KEY: str = ""
    DATABASE_URL: str = ""
    REDIS_URL: str = ""

    WORKFLOW_TIMEOUT_SECONDS: int = 45
    HEALTH_CHECK_INTERVAL_SECONDS: int = 300
    AUTO_RECOVERY_ENABLED: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

