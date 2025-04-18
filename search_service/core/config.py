from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Harena Search Service"
    API_V1_STR: str = "/api/v1"
    
    # Configuration SearchBox pour Elasticsearch
    SEARCHBOX_URL: str = os.environ.get("SEARCHBOX_URL", "")
    SEARCHBOX_API_KEY: str = os.environ.get("SEARCHBOX_API_KEY", "")
    
    # Configuration Qdrant (réutilisée depuis sync_service si possible)
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.environ.get("QDRANT_API_KEY", "")
    
    # Configuration DeepSeek
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_CHAT_MODEL: str = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_REASONER_MODEL: str = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")
    DEEPSEEK_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "8192"))
    DEEPSEEK_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_TEMPERATURE", "1.0"))
    DEEPSEEK_TOP_P: float = float(os.environ.get("DEEPSEEK_TOP_P", "0.95"))
    DEEPSEEK_TIMEOUT: int = int(os.environ.get("DEEPSEEK_TIMEOUT", "60"))
    
    # Configuration de cache en mémoire
    MEMORY_CACHE_TTL: int = 3600  # Secondes
    
    # Configuration de performances
    BATCH_SIZE: int = 32
    
    # Configuration de base de données partagée
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL")
    
    # URLs des services
    USER_SERVICE_URL: str = os.environ.get("USER_SERVICE_URL", "")
    SYNC_SERVICE_URL: str = os.environ.get("SYNC_SERVICE_URL", "")
    
    # Clé secrète pour sécuriser les jetons JWT
    SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()