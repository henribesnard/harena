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
    
    # Configuration OpenAI pour les embeddings
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Paramètres de recherche par défaut
    DEFAULT_LEXICAL_WEIGHT: float = float(os.environ.get("DEFAULT_LEXICAL_WEIGHT", "0.5"))
    DEFAULT_SEMANTIC_WEIGHT: float = float(os.environ.get("DEFAULT_SEMANTIC_WEIGHT", "0.5"))
    DEFAULT_TOP_K_INITIAL: int = int(os.environ.get("DEFAULT_TOP_K_INITIAL", "50"))
    DEFAULT_TOP_K_FINAL: int = int(os.environ.get("DEFAULT_TOP_K_FINAL", "10"))
    
    # Configuration de cache en mémoire
    MEMORY_CACHE_TTL: int = int(os.environ.get("MEMORY_CACHE_TTL", "3600"))  # Secondes
    MEMORY_CACHE_MAX_SIZE: int = int(os.environ.get("MEMORY_CACHE_MAX_SIZE", "10000"))  # Nombre d'entrées
    
    # Configuration de performances
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"))
    
    # Configuration de base de données partagée
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL")
    
    # URLs des services
    USER_SERVICE_URL: str = os.environ.get("USER_SERVICE_URL", "")
    SYNC_SERVICE_URL: str = os.environ.get("SYNC_SERVICE_URL", "")
    
    # Clé secrète pour sécuriser les jetons JWT
    SECRET_KEY: str = os.environ.get("SECRET_KEY", secrets.token_urlsafe(32))
    
    # Environnement d'exécution
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "production")


    COHERE_KEY: Optional[str] = os.environ.get("COHERE_KEY")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"
 
settings = Settings()