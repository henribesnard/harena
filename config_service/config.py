from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets
import os
from pydantic import field_validator
from urllib.parse import quote_plus
import logging

# Configurer un logger pour le module
logger = logging.getLogger(__name__)

class GlobalSettings(BaseSettings):
    """
    Configuration globale pour tous les services Harena.
    Cette classe centralise toutes les variables d'environnement utilisées par les différents services.
    """
    # Configuration générale de l'application
    PROJECT_NAME: str = "Harena Finance Platform"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 8))  # 8 jours par défaut
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "production")
    
    # Configuration base de données
    POSTGRES_SERVER: str = os.environ.get("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "harena")
    POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5432")
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL", None)
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
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
    LOG_FILE: str = os.environ.get("LOG_FILE", "harena.log")
    LOG_TO_FILE: bool = os.environ.get("LOG_TO_FILE", "False").lower() == "true"
    
    # Configuration SearchBox pour Elasticsearch
    SEARCHBOX_URL: str = os.environ.get("SEARCHBOX_URL", "")
    SEARCHBOX_API_KEY: str = os.environ.get("SEARCHBOX_API_KEY", "")
    BONSAI_URL: str = os.environ.get("BONSAI_URL", "")
    BONSAI_ACCESS_KEY : str = os.environ.get("BONSAI_ACCESS_KEY", "")
    BONSAI_SECRET_KEY : str = os.environ.get("BONSAI_SECRET_KEY", "")
    
    # Configuration Qdrant pour le stockage vectoriel
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
    
    # Configuration OpenAI pour les embeddings
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Configuration Cohere pour le reranking
    COHERE_KEY: str = os.environ.get("COHERE_KEY", "")
    
    # Configuration de performance et cache
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))
    CACHE_TTL: int = int(os.environ.get("CACHE_TTL", "3600"))
    MEMORY_CACHE_TTL: int = int(os.environ.get("MEMORY_CACHE_TTL", "3600"))
    MEMORY_CACHE_MAX_SIZE: int = int(os.environ.get("MEMORY_CACHE_MAX_SIZE", "10000"))
    
    # Configuration de taux de limite
    RATE_LIMIT_ENABLED: bool = os.environ.get("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_PERIOD: int = int(os.environ.get("RATE_LIMIT_PERIOD", "60"))
    RATE_LIMIT_REQUESTS: int = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
    
    # Paramètres de recherche par défaut
    DEFAULT_LEXICAL_WEIGHT: float = float(os.environ.get("DEFAULT_LEXICAL_WEIGHT", "0.5"))
    DEFAULT_SEMANTIC_WEIGHT: float = float(os.environ.get("DEFAULT_SEMANTIC_WEIGHT", "0.5"))
    DEFAULT_TOP_K_INITIAL: int = int(os.environ.get("DEFAULT_TOP_K_INITIAL", "50"))
    DEFAULT_TOP_K_FINAL: int = int(os.environ.get("DEFAULT_TOP_K_FINAL", "10"))
    
    # Configuration des coûts (pour le comptage de tokens)
    ENABLE_TOKEN_COUNTING: bool = os.environ.get("ENABLE_TOKEN_COUNTING", "True").lower() == "true"
    COST_PER_1K_INPUT_TOKENS: float = float(os.environ.get("COST_PER_1K_INPUT_TOKENS", "0.0005"))
    COST_PER_1K_OUTPUT_TOKENS: float = float(os.environ.get("COST_PER_1K_OUTPUT_TOKENS", "0.0015"))
    
    # Configuration de conversation
    MAX_CONVERSATION_HISTORY: int = int(os.environ.get("MAX_CONVERSATION_HISTORY", "20"))
    DEFAULT_SYSTEM_PROMPT: str = os.environ.get("DEFAULT_SYSTEM_PROMPT", 
        "Vous êtes Harena, un assistant financier intelligent qui aide les utilisateurs à comprendre et gérer leurs finances personnelles.")
    
    # URLs des services
    USER_SERVICE_URL: str = os.environ.get("USER_SERVICE_URL", "")
    SYNC_SERVICE_URL: str = os.environ.get("SYNC_SERVICE_URL", "")
    TRANSACTION_VECTOR_SERVICE_URL: str = os.environ.get("TRANSACTION_VECTOR_SERVICE_URL", "")
    
    # Configuration CORS
    CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "https://app.harena.finance")
    
    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info):
        # Utiliser directement DATABASE_URL (qui peut être DATABASE_URL dans Heroku)
        db_url = info.data.get("DATABASE_URL")
        if db_url:
            # Convertir postgres:// en postgresql:// si nécessaire
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            logger.debug("Utilisation de DATABASE_URL pour la connexion à la base de données")
            return db_url
        
        # Si pas de DATABASE_URL, utiliser les composants individuels
        data = info.data
        server = data.get("POSTGRES_SERVER", "")
        port = data.get("POSTGRES_PORT", "5432")
        user = quote_plus(data.get("POSTGRES_USER", ""))
        password = quote_plus(data.get("POSTGRES_PASSWORD", ""))
        db = data.get("POSTGRES_DB", "")
        
        if not server or server == "localhost":
            logger.debug("Configuration d'une connexion locale à la base de données")
        else:
            logger.debug(f"Configuration d'une connexion à la base de données sur {server}")
            
        return f"postgresql://{user}:{password}@{server}:{port}/{db}"
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"


# Initialisation du singleton de configuration globale
settings = GlobalSettings()

# Afficher un message de démarrage
logger.debug(f"Configuration chargée pour l'environnement: {settings.ENVIRONMENT}")