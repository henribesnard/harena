# user_service/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import secrets
from pydantic import field_validator
from urllib.parse import quote_plus

class Settings(BaseSettings):
    PROJECT_NAME: str = "Harena User Service"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 jours
    
    # Configuration base de données
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str = "5432"
    POSTGRES_URL: Optional[str] = None  
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info):
        if isinstance(v, str):
            return v
        
        # Si POSTGRES_URL est défini, utilisez-le directement
        if info.data.get("POSTGRES_URL"):
            return info.data.get("POSTGRES_URL")
        
        # Sinon, construisez l'URL à partir des composants
        data = info.data
        server = data.get("POSTGRES_SERVER", "")
        port = data.get("POSTGRES_PORT", "5432")
        user = quote_plus(data.get("POSTGRES_USER", ""))
        password = quote_plus(data.get("POSTGRES_PASSWORD", ""))
        db = data.get("POSTGRES_DB", "")
        
        return f"postgresql://{user}:{password}@{server}:{port}/{db}"
    
    # Configuration Bridge API
    BRIDGE_API_URL: str = "https://api.bridgeapi.io/v3"
    BRIDGE_API_VERSION: str = "2025-01-15"
    BRIDGE_CLIENT_ID: str
    BRIDGE_CLIENT_SECRET: str
    
    # Configuration des webhooks
    BRIDGE_WEBHOOK_SECRET: str
    WEBHOOK_BASE_URL: str = "https://api.harena.app"  # URL de base pour les webhooks
    
    # Configuration d'alertes
    ALERT_EMAIL_ENABLED: bool = False
    ALERT_EMAIL_FROM: str = "alerts@harena.app"
    ALERT_EMAIL_TO: str = "admin@harena.app"
    SMTP_SERVER: str = "smtp.example.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    
    # Configuration logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"  


settings = Settings()