# user_service/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import secrets
import os
from pydantic import field_validator
from urllib.parse import quote_plus

class Settings(BaseSettings):
    PROJECT_NAME: str = "Harena User Service"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 8))  # 8 jours par défaut
    
    # Configuration base de données
    POSTGRES_SERVER: str = os.environ.get("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "harena")
    POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5432")
    POSTGRES_URL: Optional[str] = os.environ.get("DATABASE_URL", None)
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info):
        # Utiliser directement POSTGRES_URL (qui peut être DATABASE_URL dans Heroku)
        postgres_url = info.data.get("POSTGRES_URL")
        if postgres_url:
            # Convertir postgres:// en postgresql:// si nécessaire
            if postgres_url.startswith("postgres://"):
                postgres_url = postgres_url.replace("postgres://", "postgresql://", 1)
            return postgres_url
        
        # Si pas de POSTGRES_URL, utiliser les composants individuels
        data = info.data
        server = data.get("POSTGRES_SERVER", "")
        port = data.get("POSTGRES_PORT", "5432")
        user = quote_plus(data.get("POSTGRES_USER", ""))
        password = quote_plus(data.get("POSTGRES_PASSWORD", ""))
        db = data.get("POSTGRES_DB", "")
        
        return f"postgresql://{user}:{password}@{server}:{port}/{db}"
    
    # Configuration Bridge API
    BRIDGE_API_URL: str = os.environ.get("BRIDGE_API_URL", "https://api.bridgeapi.io/v3")
    BRIDGE_API_VERSION: str = os.environ.get("BRIDGE_API_VERSION", "2025-01-15")
    BRIDGE_CLIENT_ID: str = os.environ.get("BRIDGE_CLIENT_ID", "")
    BRIDGE_CLIENT_SECRET: str = os.environ.get("BRIDGE_CLIENT_SECRET", "")
    
    # Configuration des webhooks
    BRIDGE_WEBHOOK_SECRET: str = os.environ.get("BRIDGE_WEBHOOK_SECRET", "")
    WEBHOOK_BASE_URL: str = os.environ.get("WEBHOOK_BASE_URL", "https://api.harena.app")
    
    # Configuration d'alertes avec valeurs par défaut
    ALERT_EMAIL_ENABLED: bool = os.environ.get("ALERT_EMAIL_ENABLED", "False").lower() == "true"
    ALERT_EMAIL_FROM: str = os.environ.get("ALERT_EMAIL_FROM", "alerts@harena.app")
    ALERT_EMAIL_TO: str = os.environ.get("ALERT_EMAIL_TO", "admin@harena.app")
    SMTP_SERVER: str = os.environ.get("SMTP_SERVER", "smtp.sendgrid.net")
    SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.environ.get("SMTP_USERNAME", "apikey")
    SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
    
    # Configuration logging
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"


settings = Settings()