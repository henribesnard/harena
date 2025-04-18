from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Harena Search Service"
    API_V1_STR: str = "/api/v1"
    
    # Configuration Elasticsearch
    ELASTICSEARCH_URL: str = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
    ELASTICSEARCH_USERNAME: Optional[str] = os.environ.get("ELASTICSEARCH_USERNAME")
    ELASTICSEARCH_PASSWORD: Optional[str] = os.environ.get("ELASTICSEARCH_PASSWORD")
    
    # Configuration Qdrant
    QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.environ.get("QDRANT_API_KEY")
    
    # Configuration Redis pour cache
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    
    # Configuration des embeddings
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Configuration de recherche
    DEFAULT_LEXICAL_WEIGHT: float = 0.5
    DEFAULT_SEMANTIC_WEIGHT: float = 0.5
    DEFAULT_TOP_K_INITIAL: int = 50
    DEFAULT_TOP_K_FINAL: int = 10
    
    # Configuration de performances
    BATCH_SIZE: int = 32
    CACHE_TTL: int = 3600  # Secondes
    
    # Configuration de base de données partagée
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()