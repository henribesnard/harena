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
        intent_context: Optional[IntentType] = None,
        expected_intent: Optional[IntentType] = None,
        **kwargs
    ):
        super().__init__(
            message, 
            error_code="INTENT_DETECTION_FAILED",
            **kwargs
        )
        self.query = query
        self.attempted_methods = attempted_methods or []
        self.intent_context = intent_context
        self.expected_intent = expected_intent
        self.context.update({
            "query": query,
            "attempted_methods": [str(m) for m in self.attempted_methods],
            "intent_context": intent_context.value if intent_context else None,
            "expected_intent": expected_intent.value if expected_intent else None
        })


class RuleEngineError(IntentDetectionError):
    """Erreur dans le moteur de règles"""
    
    def __init__(
        self,
        message: str,
        pattern_error: Optional[str] = None,
        failed_patterns: Optional[List[str]] = None,
        intent_type: Optional[IntentType] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="RULE_ENGINE_ERROR",
            intent_context=intent_type,
            **kwargs
        )
        self.pattern_error = pattern_error
        self.failed_patterns = failed_patterns or []
        self.intent_type = intent_type
        self.details.update({
            "pattern_error": pattern_error,
            "failed_patterns": failed_patterns,
            "intent_type": intent_type.value if intent_type else None
        })


class EntityExtractionError(ConversationServiceError):
    """Erreur dans l'extraction d'entités"""
    
    def __init__(
        self,
        message: str,
        query: str = "",
        target_entities: Optional[List[str]] = None,
        partial_entities: Optional[Dict[str, Any]] = None,
        extraction_method: str = "unknown",
        entity_type: Optional[str] = None,
        intent_context: Optional[IntentType] = None,
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
        self.extraction_method = extraction_method
        self.entity_type = entity_type
        self.intent_context = intent_context
        self.context.update({
            "query": query,
            "target_entities": target_entities,
            "partial_entities": partial_entities,
            "extraction_method": extraction_method,
            "entity_type": entity_type,
            "intent_context": intent_context.value if intent_context else None
        })


class LLMFallbackError(ConversationServiceError):
    """Erreur dans le fallback LLM (DeepSeek)"""
    
    def __init__(
        self,
        message: str,
        llm_provider: str = "deepseek",
        api_error: Optional[str] = None,
        status_code: Optional[int] = None,
        original_intent: Optional[IntentType] = None,
        fallback_attempt_count: int = 1,
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
        self.original_intent = original_intent
        self.fallback_attempt_count = fallback_attempt_count
        self.details.update({
            "llm_provider": llm_provider,
            "api_error": api_error,
            "status_code": status_code,
            "original_intent": original_intent.value if original_intent else None,
            "fallback_attempt_count": fallback_attempt_count
        })


class ConfidenceError(ConversationServiceError):
    """Erreur liée aux scores de confiance"""
    
    def __init__(
        self,
        message: str,
        confidence_score: Optional[float] = None,
        threshold: Optional[float] = None,
        method: Optional[DetectionMethod] = None,
        detected_intent: Optional[IntentType] = None,
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
        self.detected_intent = detected_intent
        self.details.update({
            "confidence_score": confidence_score,
            "threshold": threshold,
            "method": str(method) if method else None,
            "detected_intent": detected_intent.value if detected_intent else None
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
        unsupported_intent: Optional[IntentType] = None,
        supported_intents: Optional[List[IntentType]] = None,
        suggested_intent: Optional[IntentType] = None,
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
        self.suggested_intent = suggested_intent
        self.details.update({
            "unsupported_intent": unsupported_intent.value if unsupported_intent else None,
            "supported_intents": [intent.value for intent in self.supported_intents],
            "suggested_intent": suggested_intent.value if suggested_intent else None
        })


# Exceptions utilitaires pour gestion d'erreurs spécifiques
class PatternCompilationError(RuleEngineError):
    """Erreur de compilation des patterns regex"""
    
    def __init__(
        self, 
        pattern: str, 
        regex_error: str, 
        intent_type: Optional[IntentType] = None,
        **kwargs
    ):
        super().__init__(
            f"Erreur compilation pattern: {pattern}",
            pattern_error=regex_error,
            failed_patterns=[pattern],
            intent_type=intent_type,
            **kwargs
        )
        self.pattern = pattern
        self.regex_error = regex_error


class DeepSeekAPIError(LLMFallbackError):
    """Erreur spécifique API DeepSeek"""
    
    def __init__(
        self, 
        api_message: str, 
        status_code: int = 500, 
        original_intent: Optional[IntentType] = None,
        **kwargs
    ):
        super().__init__(
            f"Erreur API DeepSeek: {api_message}",
            llm_provider="deepseek",
            api_error=api_message,
            status_code=status_code,
            original_intent=original_intent,
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


class IntentMismatchError(ConversationServiceError):
    """Erreur de correspondance d'intention"""
    
    def __init__(
        self,
        message: str,
        expected_intent: IntentType,
        detected_intent: IntentType,
        confidence_score: float,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="INTENT_MISMATCH",
            severity=ErrorSeverity.WARNING,
            **kwargs
        )
        self.expected_intent = expected_intent
        self.detected_intent = detected_intent
        self.confidence_score = confidence_score
        self.details.update({
            "expected_intent": expected_intent.value,
            "detected_intent": detected_intent.value,
            "confidence_score": confidence_score,
            "intent_distance": self._calculate_intent_distance(expected_intent, detected_intent)
        })
    
    def _calculate_intent_distance(self, intent1: IntentType, intent2: IntentType) -> str:
        """Calcule une distance conceptuelle entre les intentions"""
        # Groupes d'intentions similaires
        search_group = {IntentType.SEARCH_BY_CATEGORY, IntentType.SEARCH_BY_DATE, IntentType.SEARCH_BY_MERCHANT}
        analysis_group = {IntentType.BUDGET_ANALYSIS, IntentType.ACCOUNT_BALANCE}
        action_group = {IntentType.TRANSFER, IntentType.CARD_MANAGEMENT}
        
        if intent1 == intent2:
            return "identical"
        elif (intent1 in search_group and intent2 in search_group):
            return "similar_search"
        elif (intent1 in analysis_group and intent2 in analysis_group):
            return "similar_analysis"
        elif (intent1 in action_group and intent2 in action_group):
            return "similar_action"
        else:
            return "different_category"


# Fonctions utilitaires de gestion d'erreurs
def handle_llm_error(
    exception: Exception, 
    query: str = "",
    original_intent: Optional[IntentType] = None
) -> LLMFallbackError:
    """Convertit exception LLM générique en LLMFallbackError"""
    if hasattr(exception, 'status_code'):
        status_code = getattr(exception, 'status_code')
    else:
        status_code = None
    
    return LLMFallbackError(
        message=f"Erreur LLM: {str(exception)}",
        api_error=str(exception),
        status_code=status_code,
        original_intent=original_intent,
        context={"original_query": query}
    )


def handle_validation_error(exception: Exception, field: str = "") -> ValidationError:
    """Convertit exception validation générique en ValidationError"""
    return ValidationError(
        message=f"Erreur validation: {str(exception)}",
        field_name=field,
        context={"original_exception": str(exception)}
    )


def handle_intent_detection_error(
    exception: Exception,
    query: str,
    intent_context: Optional[IntentType] = None,
    attempted_methods: Optional[List[DetectionMethod]] = None
) -> IntentDetectionError:
    """Convertit exception générique en IntentDetectionError avec contexte"""
    return IntentDetectionError(
        message=f"Erreur détection intention: {str(exception)}",
        query=query,
        intent_context=intent_context,
        attempted_methods=attempted_methods or [],
        context={"original_exception": str(exception)}
    )


def create_intent_mismatch_error(
    expected: IntentType,
    detected: IntentType, 
    confidence: float,
    query: str = ""
) -> IntentMismatchError:
    """Crée une erreur de non-correspondance d'intention"""
    return IntentMismatchError(
        message=f"Intention attendue '{expected.value}' mais détectée '{detected.value}' (confiance: {confidence:.3f})",
        expected_intent=expected,
        detected_intent=detected,
        confidence_score=confidence,
        context={"query": query}
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
    "IntentMismatchError",
    
    # Exceptions spécialisées
    "PatternCompilationError",
    "DeepSeekAPIError",
    "CacheFullError",
    
    # Fonctions utilitaires
    "handle_llm_error",
    "handle_validation_error",
    "handle_intent_detection_error",
    "create_intent_mismatch_error"
]