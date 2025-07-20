import os
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConversationSettings(BaseSettings):
    """Configuration centralisée pour le Conversation Service avec optimisations performance"""
    
    # ==========================================
    # CONFIGURATION DEEPSEEK - OPTIMISÉE PAR TÂCHE
    # ==========================================
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_CHAT_MODEL: str = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_REASONER_MODEL: str = os.environ.get("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")
    
    # Configuration Intent Classification (ultra-rapide : 100-300ms)
    DEEPSEEK_INTENT_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_INTENT_MAX_TOKENS", "100"))
    DEEPSEEK_INTENT_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_INTENT_TEMPERATURE", "0.1"))
    DEEPSEEK_INTENT_TIMEOUT: int = int(os.environ.get("DEEPSEEK_INTENT_TIMEOUT", "8"))
    DEEPSEEK_INTENT_TOP_P: float = float(os.environ.get("DEEPSEEK_INTENT_TOP_P", "0.9"))
    
    # Configuration Entity Extraction (très rapide : 50-200ms)
    DEEPSEEK_ENTITY_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_ENTITY_MAX_TOKENS", "80"))
    DEEPSEEK_ENTITY_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_ENTITY_TEMPERATURE", "0.05"))
    DEEPSEEK_ENTITY_TIMEOUT: int = int(os.environ.get("DEEPSEEK_ENTITY_TIMEOUT", "6"))
    DEEPSEEK_ENTITY_TOP_P: float = float(os.environ.get("DEEPSEEK_ENTITY_TOP_P", "0.8"))
    
    # Configuration Query Generation (rapide : 200-500ms)
    DEEPSEEK_QUERY_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_QUERY_MAX_TOKENS", "300"))
    DEEPSEEK_QUERY_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_QUERY_TEMPERATURE", "0.2"))
    DEEPSEEK_QUERY_TIMEOUT: int = int(os.environ.get("DEEPSEEK_QUERY_TIMEOUT", "10"))
    DEEPSEEK_QUERY_TOP_P: float = float(os.environ.get("DEEPSEEK_QUERY_TOP_P", "0.9"))
    
    # Configuration Response Generation (créatif mais contrôlé : 1-3s)
    DEEPSEEK_RESPONSE_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_RESPONSE_MAX_TOKENS", "500"))
    DEEPSEEK_RESPONSE_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_RESPONSE_TEMPERATURE", "0.7"))
    DEEPSEEK_RESPONSE_TIMEOUT: int = int(os.environ.get("DEEPSEEK_RESPONSE_TIMEOUT", "15"))
    DEEPSEEK_RESPONSE_TOP_P: float = float(os.environ.get("DEEPSEEK_RESPONSE_TOP_P", "0.95"))
    
    # Configuration DeepSeek Legacy (pour compatibilité)
    DEEPSEEK_MAX_TOKENS: int = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "8192"))
    DEEPSEEK_TEMPERATURE: float = float(os.environ.get("DEEPSEEK_TEMPERATURE", "1.0"))
    DEEPSEEK_TOP_P: float = float(os.environ.get("DEEPSEEK_TOP_P", "0.95"))
    DEEPSEEK_TIMEOUT: int = int(os.environ.get("DEEPSEEK_TIMEOUT", "60"))
    
    # ==========================================
    # CONFIGURATION REDIS - CACHE DISTRIBUÉ
    # ==========================================
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD: Optional[str] = os.environ.get("REDIS_PASSWORD", None)
    REDIS_DB: int = int(os.environ.get("REDIS_DB", "0"))
    REDIS_MAX_CONNECTIONS: int = int(os.environ.get("REDIS_MAX_CONNECTIONS", "20"))
    REDIS_RETRY_ON_TIMEOUT: bool = os.environ.get("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    REDIS_HEALTH_CHECK_INTERVAL: int = int(os.environ.get("REDIS_HEALTH_CHECK_INTERVAL", "30"))
    
    # ==========================================
    # CONFIGURATION CACHE MULTI-NIVEAUX
    # ==========================================
    # Cache TTL par tâche (en secondes)
    CACHE_TTL_INTENT: int = int(os.environ.get("CACHE_TTL_INTENT", "300"))  # 5 minutes
    CACHE_TTL_ENTITY: int = int(os.environ.get("CACHE_TTL_ENTITY", "180"))  # 3 minutes
    CACHE_TTL_QUERY: int = int(os.environ.get("CACHE_TTL_QUERY", "120"))    # 2 minutes
    CACHE_TTL_RESPONSE: int = int(os.environ.get("CACHE_TTL_RESPONSE", "60"))  # 1 minute
    
    # Cache L1 - Mémoire locale
    MEMORY_CACHE_SIZE: int = int(os.environ.get("MEMORY_CACHE_SIZE", "2000"))
    MEMORY_CACHE_TTL: int = int(os.environ.get("MEMORY_CACHE_TTL", "300"))
    
    # Cache L0 - Patterns pré-calculés
    PRECOMPUTED_PATTERNS_ENABLED: bool = os.environ.get("PRECOMPUTED_PATTERNS_ENABLED", "true").lower() == "true"
    PRECOMPUTED_PATTERNS_SIZE: int = int(os.environ.get("PRECOMPUTED_PATTERNS_SIZE", "100"))
    
    # Cache L2 - Redis distribué
    REDIS_CACHE_ENABLED: bool = os.environ.get("REDIS_CACHE_ENABLED", "true").lower() == "true"
    REDIS_CACHE_PREFIX: str = os.environ.get("REDIS_CACHE_PREFIX", "conversation_service")
    
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
    CLASSIFICATION_CACHE_TTL: int = int(os.environ.get("CLASSIFICATION_CACHE_TTL", "300"))  # Legacy
    CACHE_SIZE: int = int(os.environ.get("CACHE_SIZE", "1000"))  # Legacy
    
    # ==========================================
    # CONFIGURATION API
    # ==========================================
    API_TITLE: str = "Conversation Service"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Service de classification d'intentions financières optimisé"
    
    # ==========================================
    # CONFIGURATION PERFORMANCE OPTIMISÉE
    # ==========================================
    REQUEST_TIMEOUT: int = int(os.environ.get("REQUEST_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.environ.get("RETRY_DELAY", "1.0"))
    
    # Optimisations pipeline asynchrone
    ENABLE_ASYNC_PIPELINE: bool = os.environ.get("ENABLE_ASYNC_PIPELINE", "true").lower() == "true"
    PIPELINE_TIMEOUT: int = int(os.environ.get("PIPELINE_TIMEOUT", "10"))
    PARALLEL_PROCESSING_ENABLED: bool = os.environ.get("PARALLEL_PROCESSING_ENABLED", "true").lower() == "true"
    
    # Thread pools pour traitement asynchrone
    THREAD_POOL_SIZE: int = int(os.environ.get("THREAD_POOL_SIZE", "10"))
    MAX_CONCURRENT_REQUESTS: int = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "100"))
    
    # ==========================================
    # CONFIGURATION MONITORING AVANCÉ
    # ==========================================
    ENABLE_METRICS: bool = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
    METRICS_ENDPOINT: str = "/metrics"
    HEALTH_ENDPOINT: str = "/health"
    
    # Métriques performance détaillées
    ENABLE_DETAILED_METRICS: bool = os.environ.get("ENABLE_DETAILED_METRICS", "true").lower() == "true"
    METRICS_COLLECTION_INTERVAL: int = int(os.environ.get("METRICS_COLLECTION_INTERVAL", "60"))
    
    # Alertes performance
    PERFORMANCE_ALERT_THRESHOLD_MS: int = int(os.environ.get("PERFORMANCE_ALERT_THRESHOLD_MS", "2000"))
    CACHE_HIT_RATE_ALERT_THRESHOLD: float = float(os.environ.get("CACHE_HIT_RATE_ALERT_THRESHOLD", "0.8"))
    ERROR_RATE_ALERT_THRESHOLD: float = float(os.environ.get("ERROR_RATE_ALERT_THRESHOLD", "0.05"))
    
    # ==========================================
    # CONFIGURATION CIRCUIT BREAKER
    # ==========================================
    CIRCUIT_BREAKER_ENABLED: bool = os.environ.get("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(os.environ.get("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = int(os.environ.get("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60"))
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = os.environ.get("CIRCUIT_BREAKER_EXPECTED_EXCEPTION", "httpx.RequestError")
    
    # ==========================================
    # CONFIGURATION RATE LIMITING
    # ==========================================
    RATE_LIMIT_ENABLED: bool = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
    RATE_LIMIT_BURST_SIZE: int = int(os.environ.get("RATE_LIMIT_BURST_SIZE", "10"))
    
    # ==========================================
    # CONFIGURATION BATCH PROCESSING
    # ==========================================
    BATCH_PROCESSING_ENABLED: bool = os.environ.get("BATCH_PROCESSING_ENABLED", "true").lower() == "true"
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "5"))
    BATCH_TIMEOUT_MS: int = int(os.environ.get("BATCH_TIMEOUT_MS", "100"))
    
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
        
        # Validation Redis (si activé)
        if self.REDIS_CACHE_ENABLED:
            if not self.REDIS_URL:
                errors.append("REDIS_URL est requis quand REDIS_CACHE_ENABLED=true")
        
        # Validation thresholds
        if self.MIN_CONFIDENCE_THRESHOLD < 0.0 or self.MIN_CONFIDENCE_THRESHOLD > 1.0:
            errors.append("MIN_CONFIDENCE_THRESHOLD doit être entre 0.0 et 1.0")
        
        if self.MIN_CONFIDENCE_THRESHOLD < 0.5:
            warnings.append("MIN_CONFIDENCE_THRESHOLD < 0.5 peut produire des résultats peu fiables")
        
        # Validation performance
        if self.REQUEST_TIMEOUT < 5:
            warnings.append("REQUEST_TIMEOUT < 5s peut causer des timeouts")
        
        if self.MEMORY_CACHE_SIZE < 100:
            warnings.append("MEMORY_CACHE_SIZE < 100 peut réduire les performances")
        
        # Validation timeouts optimisés
        if self.DEEPSEEK_INTENT_TIMEOUT > 10:
            warnings.append("DEEPSEEK_INTENT_TIMEOUT > 10s est trop élevé pour l'optimisation")
        
        if self.DEEPSEEK_INTENT_MAX_TOKENS > 150:
            warnings.append("DEEPSEEK_INTENT_MAX_TOKENS > 150 peut ralentir la classification")
        
        # Validation cache TTL
        if self.CACHE_TTL_INTENT < 60:
            warnings.append("CACHE_TTL_INTENT < 60s peut réduire l'efficacité du cache")
        
        # Validation rate limiting
        if self.RATE_LIMIT_REQUESTS_PER_MINUTE < 10:
            warnings.append("RATE_LIMIT_REQUESTS_PER_MINUTE très bas, peut impacter l'utilisabilité")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def get_deepseek_config(self, task_type: str = "default") -> dict:
        """Retourne la configuration DeepSeek optimisée par tâche"""
        
        configs = {
            "intent": {
                "api_key": self.DEEPSEEK_API_KEY,
                "base_url": self.DEEPSEEK_BASE_URL,
                "chat_model": self.DEEPSEEK_CHAT_MODEL,
                "max_tokens": self.DEEPSEEK_INTENT_MAX_TOKENS,
                "temperature": self.DEEPSEEK_INTENT_TEMPERATURE,
                "top_p": self.DEEPSEEK_INTENT_TOP_P,
                "timeout": self.DEEPSEEK_INTENT_TIMEOUT
            },
            "entity": {
                "api_key": self.DEEPSEEK_API_KEY,
                "base_url": self.DEEPSEEK_BASE_URL,
                "chat_model": self.DEEPSEEK_CHAT_MODEL,
                "max_tokens": self.DEEPSEEK_ENTITY_MAX_TOKENS,
                "temperature": self.DEEPSEEK_ENTITY_TEMPERATURE,
                "top_p": self.DEEPSEEK_ENTITY_TOP_P,
                "timeout": self.DEEPSEEK_ENTITY_TIMEOUT
            },
            "query": {
                "api_key": self.DEEPSEEK_API_KEY,
                "base_url": self.DEEPSEEK_BASE_URL,
                "chat_model": self.DEEPSEEK_CHAT_MODEL,
                "max_tokens": self.DEEPSEEK_QUERY_MAX_TOKENS,
                "temperature": self.DEEPSEEK_QUERY_TEMPERATURE,
                "top_p": self.DEEPSEEK_QUERY_TOP_P,
                "timeout": self.DEEPSEEK_QUERY_TIMEOUT
            },
            "response": {
                "api_key": self.DEEPSEEK_API_KEY,
                "base_url": self.DEEPSEEK_BASE_URL,
                "chat_model": self.DEEPSEEK_CHAT_MODEL,
                "max_tokens": self.DEEPSEEK_RESPONSE_MAX_TOKENS,
                "temperature": self.DEEPSEEK_RESPONSE_TEMPERATURE,
                "top_p": self.DEEPSEEK_RESPONSE_TOP_P,
                "timeout": self.DEEPSEEK_RESPONSE_TIMEOUT
            },
            "default": {
                "api_key": self.DEEPSEEK_API_KEY,
                "base_url": self.DEEPSEEK_BASE_URL,
                "chat_model": self.DEEPSEEK_CHAT_MODEL,
                "reasoner_model": self.DEEPSEEK_REASONER_MODEL,
                "max_tokens": self.DEEPSEEK_MAX_TOKENS,
                "temperature": self.DEEPSEEK_TEMPERATURE,
                "top_p": self.DEEPSEEK_TOP_P,
                "timeout": self.DEEPSEEK_TIMEOUT
            }
        }
        
        return configs.get(task_type, configs["default"])
    
    def get_cache_config(self) -> dict:
        """Retourne la configuration cache multi-niveaux"""
        return {
            # Redis configuration
            "redis": {
                "enabled": self.REDIS_CACHE_ENABLED,
                "url": self.REDIS_URL,
                "password": self.REDIS_PASSWORD,
                "db": self.REDIS_DB,
                "max_connections": self.REDIS_MAX_CONNECTIONS,
                "retry_on_timeout": self.REDIS_RETRY_ON_TIMEOUT,
                "health_check_interval": self.REDIS_HEALTH_CHECK_INTERVAL,
                "prefix": self.REDIS_CACHE_PREFIX
            },
            # Memory cache L1
            "memory": {
                "size": self.MEMORY_CACHE_SIZE,
                "ttl": self.MEMORY_CACHE_TTL
            },
            # Precomputed patterns L0
            "precomputed": {
                "enabled": self.PRECOMPUTED_PATTERNS_ENABLED,
                "size": self.PRECOMPUTED_PATTERNS_SIZE
            },
            # TTL par tâche
            "ttl": {
                "intent": self.CACHE_TTL_INTENT,
                "entity": self.CACHE_TTL_ENTITY,
                "query": self.CACHE_TTL_QUERY,
                "response": self.CACHE_TTL_RESPONSE
            }
        }
    
    def get_performance_config(self) -> dict:
        """Retourne la configuration de performance optimisée"""
        return {
            "request_timeout": self.REQUEST_TIMEOUT,
            "max_retries": self.MAX_RETRIES,
            "retry_delay": self.RETRY_DELAY,
            "cache_size": self.MEMORY_CACHE_SIZE,
            "cache_ttl": self.MEMORY_CACHE_TTL,
            "confidence_threshold": self.MIN_CONFIDENCE_THRESHOLD,
            
            # Pipeline asynchrone
            "async_pipeline": self.ENABLE_ASYNC_PIPELINE,
            "pipeline_timeout": self.PIPELINE_TIMEOUT,
            "parallel_processing": self.PARALLEL_PROCESSING_ENABLED,
            "thread_pool_size": self.THREAD_POOL_SIZE,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            
            # Circuit breaker
            "circuit_breaker": {
                "enabled": self.CIRCUIT_BREAKER_ENABLED,
                "failure_threshold": self.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                "recovery_timeout": self.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
            },
            
            # Rate limiting
            "rate_limit": {
                "enabled": self.RATE_LIMIT_ENABLED,
                "requests_per_minute": self.RATE_LIMIT_REQUESTS_PER_MINUTE,
                "burst_size": self.RATE_LIMIT_BURST_SIZE
            },
            
            # Batch processing
            "batch": {
                "enabled": self.BATCH_PROCESSING_ENABLED,
                "size": self.BATCH_SIZE,
                "timeout_ms": self.BATCH_TIMEOUT_MS
            }
        }
    
    def get_monitoring_config(self) -> dict:
        """Retourne la configuration de monitoring"""
        return {
            "enabled": self.ENABLE_METRICS,
            "detailed": self.ENABLE_DETAILED_METRICS,
            "collection_interval": self.METRICS_COLLECTION_INTERVAL,
            "endpoints": {
                "metrics": self.METRICS_ENDPOINT,
                "health": self.HEALTH_ENDPOINT
            },
            "alerts": {
                "performance_threshold_ms": self.PERFORMANCE_ALERT_THRESHOLD_MS,
                "cache_hit_rate_threshold": self.CACHE_HIT_RATE_ALERT_THRESHOLD,
                "error_rate_threshold": self.ERROR_RATE_ALERT_THRESHOLD
            }
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
logger.info(f"Cache Redis: {'Activé' if settings.REDIS_CACHE_ENABLED else 'Désactivé'}")
logger.info(f"Pipeline Asynchrone: {'Activé' if settings.ENABLE_ASYNC_PIPELINE else 'Désactivé'}")