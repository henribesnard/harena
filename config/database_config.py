from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from urllib.parse import quote_plus


class DatabaseSettings(BaseSettings):
    """Paramètres de configuration pour la base de données."""

    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = ""
    POSTGRES_PORT: int = 5432
    SQLALCHEMY_DATABASE_URI: str | None = None

    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str | None, info):
        if v:
            return v
        data = info.data
        user = quote_plus(data.get("POSTGRES_USER", "postgres"))
        password = quote_plus(data.get("POSTGRES_PASSWORD", ""))
        server = data.get("POSTGRES_SERVER", "localhost")
        port = data.get("POSTGRES_PORT", 5432)
        db = data.get("POSTGRES_DB", "")
        return f"postgresql://{user}:{password}@{server}:{port}/{db}"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
