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
    
    # Vectorisation
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384
    
    # Tâches planifiées
    ENABLE_SCHEDULED_TASKS: bool = os.getenv("ENABLE_SCHEDULED_TASKS", "True").lower() == "true"
    INSIGHTS_GENERATION_CRON: str = os.getenv("INSIGHTS_GENERATION_CRON", "0 3 * * *")  # 3h du matin tous les jours
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()