"""
Configuration settings for conversation_service_v3
"""
import os
from typing import Optional, List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings


def clean_env_value(value: str) -> str:
    """Remove quotes and whitespace from environment variables"""
    if not value:
        return value
    # Remove surrounding quotes (single and double) and whitespace
    return value.strip().strip("'").strip('"').strip()


class Settings(BaseSettings):
    """Application settings"""

    # Service info
    SERVICE_NAME: str = "conversation_service_v3"
    SERVICE_VERSION: str = "3.0.0"
    API_V3_PREFIX: str = "/api/v3"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 3008

    # External services
    SEARCH_SERVICE_URL: str = "http://localhost:3002"
    BUDGET_SERVICE_URL: str = "http://harena_budget_profiling_service:3006"

    # Budget Profile Integration
    BUDGET_PROFILE_ENABLED: bool = True  # Feature flag
    BUDGET_PROFILE_TIMEOUT: float = 5.0  # Timeout en secondes

    # LLM Configuration
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_RESPONSE_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.1

    # Agent configuration
    MAX_CORRECTION_ATTEMPTS: int = 2
    QUERY_TIMEOUT_SECONDS: int = 30

    # Context optimization - Limite le nombre de transactions dans le contexte LLM
    MAX_TRANSACTIONS_IN_CONTEXT: int = 50

    # Redis Configuration - Conversation Memory Cache
    REDIS_URL: str = "redis://:HaReNa2024-Redis-Auth-Token-Secure-Key-123456@63.35.52.216:6379/0"
    REDIS_CONVERSATION_CACHE_ENABLED: bool = True

    # Validators pour nettoyer les guillemets du .env
    @field_validator('SEARCH_SERVICE_URL', 'BUDGET_SERVICE_URL', 'REDIS_URL', 'OPENAI_API_KEY', 'LLM_MODEL', 'LLM_RESPONSE_MODEL', mode='before')
    @classmethod
    def clean_string_fields(cls, v):
        """Remove quotes from string fields loaded from .env"""
        if isinstance(v, str):
            return clean_env_value(v)
        return v

    # Conversation Memory Limits
    MAX_CONVERSATION_MESSAGES: int = int(os.getenv("MAX_CONVERSATION_MESSAGES", "10"))  # Sliding window size
    MAX_CONVERSATION_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONVERSATION_CONTEXT_TOKENS", "4000"))  # Token limit for history
    CONVERSATION_CACHE_TTL_SECONDS: int = int(os.getenv("CONVERSATION_CACHE_TTL_SECONDS", "86400"))  # 24 hours

    # CORS - Will be parsed manually
    CORS_ORIGINS: str = clean_env_value(os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5174"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # JWT Configuration (compatible user_service)
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")

    @field_validator('SECRET_KEY', mode='before')
    @classmethod
    def validate_secret_key(cls, v):
        """Validate and clean SECRET_KEY"""
        if isinstance(v, str):
            v = clean_env_value(v)
        if not v or len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from string"""
        if isinstance(self.CORS_ORIGINS, list):
            return self.CORS_ORIGINS
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    class Config:
        case_sensitive = True
        env_file = ".env"


# Instance globale
settings = Settings()
