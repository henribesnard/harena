"""
❌ Exceptions Métier - Gestion d'erreurs spécialisées

Exceptions spécifiques au domaine de détection d'intention financière.
Hiérarchie d'exceptions avec codes d'erreur et contexte détaillé.
"""

from typing import Dict, Any, Optional, List
from .enums import ErrorSeverity, DetectionMethod, IntentType


class ConversationServiceError(Exception):
    """Exception de base pour le service de conversation"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "GENERAL_ERROR",
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'exception pour logging/API"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": str(self.severity),
            "details": self.details,
            "context": self.context
        }


class IntentDetectionError(ConversationServiceError):
    """Erreur dans le processus de détection d'intention"""
    
    def __init__(
        self,
        message: str,
        query: str = "",
        attempted_methods: Optional[List[DetectionMethod]] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            error_code="INTENT_DETECTION_FAILED",
            **kwargs
        )
        self.query = query
        self.attempted_methods = attempted_methods or []
        self.context.update({
            "query": query,
            "attempted_methods": [str(m) for m in self.attempted_methods]
        })


class RuleEngineError(IntentDetectionError):
    """Erreur dans le moteur de règles"""
    
    def __init__(
        self,
        message: str,
        pattern_error: Optional[str] = None,
        failed_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="RULE_ENGINE_ERROR",
            **kwargs
        )
        self.pattern_error = pattern_error
        self.failed_patterns = failed_patterns or []
        self.details.update({
            "pattern_error": pattern_error,
            "failed_patterns": failed_patterns
        })


class EntityExtractionError(ConversationServiceError):
    """Erreur dans l'extraction d'entités"""
    
    def __init__(
        self,
        message: str,
        query: str = "",
        target_entities: Optional[List[str]] = None,
        partial_entities: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="ENTITY_EXTRACTION_FAILED", 
            **kwargs
        )
        self.query = query
        self.target_entities = target_entities or []
        self.partial_entities = partial_entities or {}
        self.context.update({
            "query": query,
            "target_entities": target_entities,
            "partial_entities": partial_entities
        })


class LLMFallbackError(ConversationServiceError):
    """Erreur dans le fallback LLM (DeepSeek)"""
    
    def __init__(
        self,
        message: str,
        llm_provider: str = "deepseek",
        api_error: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="LLM_FALLBACK_FAILED",
            **kwargs
        )
        self.llm_provider = llm_provider
        self.api_error = api_error
        self.status_code = status_code
        self.details.update({
            "llm_provider": llm_provider,
            "api_error": api_error,
            "status_code": status_code
        })


class ConfidenceError(ConversationServiceError):
    """Erreur liée aux scores de confiance"""
    
    def __init__(
        self,
        message: str,
        confidence_score: Optional[float] = None,
        threshold: Optional[float] = None,
        method: Optional[DetectionMethod] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="CONFIDENCE_ERROR",
            **kwargs
        )
        self.confidence_score = confidence_score
        self.threshold = threshold
        self.method = method
        self.details.update({
            "confidence_score": confidence_score,
            "threshold": threshold,
            "method": str(method) if method else None
        })


class CacheError(ConversationServiceError):
    """Erreur dans le système de cache"""
    
    def __init__(
        self,
        message: str,
        cache_operation: str = "unknown",
        cache_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="CACHE_ERROR",
            severity=ErrorSeverity.WARNING,  # Cache errors are usually non-critical
            **kwargs
        )
        self.cache_operation = cache_operation
        self.cache_key = cache_key
        self.details.update({
            "cache_operation": cache_operation,
            "cache_key": cache_key
        })


class ValidationError(ConversationServiceError):
    """Erreur de validation des données d'entrée"""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.WARNING,
            **kwargs
        )
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule
        self.details.update({
            "field_name": field_name,
            "field_value": str(field_value) if field_value is not None else None,
            "validation_rule": validation_rule
        })


class ConfigurationError(ConversationServiceError):
    """Erreur de configuration du service"""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        missing_keys: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.config_section = config_section
        self.missing_keys = missing_keys or []
        self.details.update({
            "config_section": config_section,
            "missing_keys": missing_keys
        })


class ServiceUnavailableError(ConversationServiceError):
    """Service temporairement indisponible"""
    
    def __init__(
        self,
        message: str,
        service_name: str = "conversation-service",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="SERVICE_UNAVAILABLE",
            severity=ErrorSeverity.ERROR,
            **kwargs
        )
        self.service_name = service_name
        self.retry_after = retry_after
        self.details.update({
            "service_name": service_name,
            "retry_after": retry_after
        })


class TimeoutError(ConversationServiceError):
    """Timeout dans le traitement"""
    
    def __init__(
        self,
        message: str,
        operation: str = "unknown",
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            **kwargs
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.details.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds
        })


class UnsupportedIntentError(ConversationServiceError):
    """Intention non supportée"""
    
    def __init__(
        self,
        message: str,
        unsupported_intent: Optional[str] = None,
        supported_intents: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="UNSUPPORTED_INTENT",
            severity=ErrorSeverity.WARNING,
            **kwargs
        )
        self.unsupported_intent = unsupported_intent
        self.supported_intents = supported_intents or []
        self.details.update({
            "unsupported_intent": unsupported_intent,
            "supported_intents": supported_intents
        })


# Exceptions utilitaires pour gestion d'erreurs spécifiques
class PatternCompilationError(RuleEngineError):
    """Erreur de compilation des patterns regex"""
    
    def __init__(self, pattern: str, regex_error: str, **kwargs):
        super().__init__(
            f"Erreur compilation pattern: {pattern}",
            pattern_error=regex_error,
            failed_patterns=[pattern],
            **kwargs
        )
        self.pattern = pattern
        self.regex_error = regex_error


class DeepSeekAPIError(LLMFallbackError):
    """Erreur spécifique API DeepSeek"""
    
    def __init__(self, api_message: str, status_code: int = 500, **kwargs):
        super().__init__(
            f"Erreur API DeepSeek: {api_message}",
            llm_provider="deepseek",
            api_error=api_message,
            status_code=status_code,
            **kwargs
        )


class CacheFullError(CacheError):
    """Cache plein, impossibilité d'ajouter"""
    
    def __init__(self, cache_size: int, max_size: int, **kwargs):
        super().__init__(
            f"Cache plein ({cache_size}/{max_size})",
            cache_operation="insert",
            **kwargs
        )
        self.cache_size = cache_size
        self.max_size = max_size
        self.details.update({
            "cache_size": cache_size,
            "max_size": max_size
        })


# Fonctions utilitaires de gestion d'erreurs
def handle_llm_error(exception: Exception, query: str = "") -> LLMFallbackError:
    """Convertit exception LLM générique en LLMFallbackError"""
    if hasattr(exception, 'status_code'):
        status_code = getattr(exception, 'status_code')
    else:
        status_code = None
    
    return LLMFallbackError(
        message=f"Erreur LLM: {str(exception)}",
        api_error=str(exception),
        status_code=status_code,
        context={"original_query": query}
    )


def handle_validation_error(exception: Exception, field: str = "") -> ValidationError:
    """Convertit exception validation générique en ValidationError"""
    return ValidationError(
        message=f"Erreur validation: {str(exception)}",
        field_name=field,
        context={"original_exception": str(exception)}
    )


# Exports publics
__all__ = [
    # Exception de base
    "ConversationServiceError",
    
    # Exceptions principales
    "IntentDetectionError",
    "RuleEngineError", 
    "EntityExtractionError",
    "LLMFallbackError",
    "ConfidenceError",
    "CacheError",
    "ValidationError",
    "ConfigurationError",
    "ServiceUnavailableError",
    "TimeoutError",
    "UnsupportedIntentError",
    
    # Exceptions spécialisées
    "PatternCompilationError",
    "DeepSeekAPIError",
    "CacheFullError",
    
    # Fonctions utilitaires
    "handle_llm_error",
    "handle_validation_error"
]