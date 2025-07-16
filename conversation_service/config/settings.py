import os
from pydantic_settings import BaseSettings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ConversationSettings(BaseSettings):
    """Configuration centralisée pour le Conversation Service"""
    
    # ==========================================
    # CONFIGURATION DEEPSEEK
    # ==========================================
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_CHAT_MODEL: str = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_REASONER_MODEL: str = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")
    DEEPSEEK_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "8192"))
    DEEPSEEK_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_TEMPERATURE", "1.0"))
    DEEPSEEK_TOP_P: float = float(os.environ.get("DEEPSEEK_TOP_P", "0.95"))
    DEEPSEEK_TIMEOUT: int = int(os.environ.get("DEEPSEEK_TIMEOUT", "60"))
    
    # ==========================================
    # CONFIGURATION SERVICE
    # ==========================================
    HOST: str = os.environ.get("CONVERSATION_SERVICE_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("CONVERSATION_SERVICE_PORT", "8001"))
    DEBUG: bool = os.environ.get("CONVERSATION_SERVICE_DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.environ.get("CONVERSATION_SERVICE_LOG_LEVEL", "INFO")
    
    # ==========================================
    # CONFIGURATION INTENT CLASSIFIER
    # ==========================================
    MIN_CONFIDENCE_THRESHOLD: float = float(os.environ.get("MIN_CONFIDENCE_THRESHOLD", "0.7"))
    CLASSIFICATION_CACHE_TTL: int = int(os.environ.get("CLASSIFICATION_CACHE_TTL", "300"))  # 5 minutes
    CACHE_SIZE: int = int(os.environ.get("CACHE_SIZE", "1000"))
    
    # ==========================================
    # CONFIGURATION API
    # ==========================================
    API_TITLE: str = "Conversation Service"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Service de classification d'intentions financières"
    
    # ==========================================
    # CONFIGURATION PERFORMANCE
    # ==========================================
    REQUEST_TIMEOUT: int = int(os.environ.get("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.environ.get("RETRY_DELAY", "1.0"))
    
    # ==========================================
    # CONFIGURATION MONITORING
    # ==========================================
    ENABLE_METRICS: bool = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
    METRICS_ENDPOINT: str = "/metrics"
    HEALTH_ENDPOINT: str = "/health"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
    
    def validate_configuration(self) -> dict:
        """Valide la configuration et retourne les erreurs"""
        errors = []
        warnings = []
        
        # Validation DeepSeek
        if not self.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY est requis")
        
        if not self.DEEPSEEK_BASE_URL:
            errors.append("DEEPSEEK_BASE_URL est requis")
        
        # Validation thresholds
        if self.MIN_CONFIDENCE_THRESHOLD < 0.0 or self.MIN_CONFIDENCE_THRESHOLD > 1.0:
            errors.append("MIN_CONFIDENCE_THRESHOLD doit être entre 0.0 et 1.0")
        
        if self.MIN_CONFIDENCE_THRESHOLD < 0.5:
            warnings.append("MIN_CONFIDENCE_THRESHOLD < 0.5 peut produire des résultats peu fiables")
        
        # Validation performance
        if self.REQUEST_TIMEOUT < 5:
            warnings.append("REQUEST_TIMEOUT < 5s peut causer des timeouts")
        
        if self.CACHE_SIZE < 100:
            warnings.append("CACHE_SIZE < 100 peut réduire les performances")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def get_deepseek_config(self) -> dict:
        """Retourne la configuration DeepSeek formatée"""
        return {
            "api_key": self.DEEPSEEK_API_KEY,
            "base_url": self.DEEPSEEK_BASE_URL,
            "chat_model": self.DEEPSEEK_CHAT_MODEL,
            "reasoner_model": self.DEEPSEEK_REASONER_MODEL,
            "max_tokens": self.DEEPSEEK_MAX_TOKENS,
            "temperature": self.DEEPSEEK_TEMPERATURE,
            "top_p": self.DEEPSEEK_TOP_P,
            "timeout": self.DEEPSEEK_TIMEOUT
        }
    
    def get_performance_config(self) -> dict:
        """Retourne la configuration de performance"""
        return {
            "request_timeout": self.REQUEST_TIMEOUT,
            "max_retries": self.MAX_RETRIES,
            "retry_delay": self.RETRY_DELAY,
            "cache_size": self.CACHE_SIZE,
            "cache_ttl": self.CLASSIFICATION_CACHE_TTL,
            "confidence_threshold": self.MIN_CONFIDENCE_THRESHOLD
        }

# Instance globale des settings
settings = ConversationSettings()

# Validation automatique au démarrage
validation_result = settings.validate_configuration()
if not validation_result["valid"]:
    logger.error(f"Configuration invalide: {validation_result['errors']}")
    raise ValueError(f"Configuration invalide: {validation_result['errors']}")

if validation_result["warnings"]:
    logger.warning(f"Avertissements configuration: {validation_result['warnings']}")

logger.info(f"Configuration Conversation Service chargée - Mode: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")
logger.info(f"Port: {settings.PORT}, Timeout: {settings.REQUEST_TIMEOUT}s, Confidence: {settings.MIN_CONFIDENCE_THRESHOLD}")