# search_service/config/settings.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configuration pour le Search Service basée sur le code existant"""
    
    # Elasticsearch - Utilisation de BONSAI_URL comme dans le code existant
    BONSAI_URL: str = os.environ.get("BONSAI_URL", "")
    ELASTICSEARCH_INDEX: str = os.environ.get("ELASTICSEARCH_INDEX", "harena_transactions")
    
    # User de test (comme dans le code existant)
    test_user_id: int = int(os.environ.get("TEST_USER_ID", "34"))
    
    # API Configuration
    api_title: str = "Search Service"
    api_version: str = "1.0.0"
    api_description: str = "Service de recherche simplifié pour les transactions"
    
    # Performance
    default_limit: int = 20
    max_limit: int = 100
    default_timeout_ms: int = 5000
    
    # Sécurité
    require_user_id: bool = True
    
    # Debug
    debug_mode: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Instance globale des settings
settings = Settings()