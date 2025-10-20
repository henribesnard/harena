"""
Configuration settings for conversation_service_v3
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


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
    SEARCH_SERVICE_URL: str = os.getenv("SEARCH_SERVICE_URL", "http://localhost:3002")

    # LLM Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_RESPONSE_MODEL: str = os.getenv("LLM_RESPONSE_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Agent configuration
    MAX_CORRECTION_ATTEMPTS: int = int(os.getenv("MAX_CORRECTION_ATTEMPTS", "2"))
    QUERY_TIMEOUT_SECONDS: int = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
    ]

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        case_sensitive = True
        env_file = ".env"


# Instance globale
settings = Settings()
