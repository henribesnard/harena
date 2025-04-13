"""
Configuration et paramètres du service de conversation.

Ce module définit la configuration utilisée par le service de conversation,
chargée depuis les variables d'environnement et les fichiers de configuration.
"""

import os
import secrets
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration du service de conversation.
    
    Cette classe utilise Pydantic pour charger et valider la configuration
    depuis les variables d'environnement et les fichiers .env.
    """
    # Informations générales
    PROJECT_NAME: str = "Harena Conversation Service"
    API_VERSION: str = "v1"
    DEBUG: bool = False
    SECRET_KEY: str = secrets.token_urlsafe(32)
    
    # Configuration du serveur
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    RELOAD: bool = False
    WORKERS: int = 1
    
    # Chemins et URLs
    API_PREFIX: str = "/api/v1"
    BASE_PATH: Path = Path(__file__).parent.parent.resolve()
    STATIC_DIR: Path = BASE_PATH / "static"
    
    # Base de données
    DATABASE_URI: str = "postgresql://postgres:postgres@localhost:5432/harena"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO_SQL: bool = False
    
    # Configuration de Deepseek LLM
    DEEPSEEK_API_KEY: str
    DEEPSEEK_BASE_URL: str
    DEEPSEEK_MODEL: str
    DEEPSEEK_TEMPERATURE: float
    DEEPSEEK_MAX_TOKENS: int
    DEEPSEEK_TOP_P: float
    DEEPSEEK_TIMEOUT: int
    
    # Paramètres de services externes
    TRANSACTION_VECTOR_SERVICE_URL: str
    SYNC_SERVICE_URL: str
    USER_SERVICE_URL: str
    
    # Paramètres de conversation
    MAX_CONVERSATION_HISTORY: int
    MAX_CONVERSATION_AGE_DAYS: int = 30
    DEFAULT_SYSTEM_PROMPT: str
    
    # Configuration des tokens pour le suivi des coûts
    ENABLE_TOKEN_COUNTING: bool = True
    COST_PER_1K_INPUT_TOKENS: float
    COST_PER_1K_OUTPUT_TOKENS: float
    
    # Configuration du cache
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # secondes
    
    # Configuration de logging
    LOG_LEVEL: str
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_TO_FILE: bool = False
    LOG_FILE: str = "conversation_service.log"
    
    # Limites de rate
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int
    RATE_LIMIT_PERIOD: int
    
    # Configuration du client model
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Méthodes de validation personnalisées
    def get_database_connection_args(self) -> Dict[str, Any]:
        """
        Obtenir les arguments de connexion à la base de données.
        
        Returns:
            Dict[str, Any]: Arguments de connexion à la base de données
        """
        return {
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "echo": self.DATABASE_ECHO_SQL
        }
    
    def get_deepseek_config(self) -> Dict[str, Any]:
        """
        Obtenir la configuration pour Deepseek LLM.
        
        Returns:
            Dict[str, Any]: Configuration pour Deepseek LLM
        """
        return {
            "api_key": self.DEEPSEEK_API_KEY,
            "base_url": self.DEEPSEEK_BASE_URL,
            "model": self.DEEPSEEK_MODEL,
            "temperature": self.DEEPSEEK_TEMPERATURE,
            "max_tokens": self.DEEPSEEK_MAX_TOKENS,
            "top_p": self.DEEPSEEK_TOP_P,
            "timeout": self.DEEPSEEK_TIMEOUT
        }


# Création d'une instance globale de configuration
settings = Settings()