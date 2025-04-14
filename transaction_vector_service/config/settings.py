# transaction_vector_service/config/settings.py
from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class Settings(BaseSettings):
    # Général
    PROJECT_NAME: str = "Harena Transaction Vector Service"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Bridge API
    BRIDGE_API_URL: str = "https://api.bridgeapi.io/v3"
    BRIDGE_API_VERSION: str = "2025-01-15"
    BRIDGE_CLIENT_ID: str = os.getenv("BRIDGE_CLIENT_ID", "")
    BRIDGE_CLIENT_SECRET: str = os.getenv("BRIDGE_CLIENT_SECRET", "")
    BRIDGE_WEBHOOK_SECRET: str = os.getenv("BRIDGE_WEBHOOK_SECRET", "")
    
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Vectorisation
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSION: int = 1536  # Dimension for text-embedding-3-small
    
    # Tâches planifiées
    ENABLE_SCHEDULED_TASKS: bool = os.getenv("ENABLE_SCHEDULED_TASKS", "True").lower() == "true"
    INSIGHTS_GENERATION_CRON: str = os.getenv("INSIGHTS_GENERATION_CRON", "0 3 * * *")  # 3h du matin tous les jours
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # JWT settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    JWT_ALGORITHM: str = "HS256"
    
    # Paramètres additionnels pour permettre la compatibilité avec d'autres services
    model_config = {
        "extra": "ignore",  # Permet d'ignorer les variables d'environnement non définies dans cette classe
        "env_file": ".env",
        "case_sensitive": True
    }

settings = Settings()