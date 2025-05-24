"""
Exceptions personnalisées pour le service d'enrichissement.

Ce module définit les exceptions spécifiques au service d'enrichissement
pour une gestion d'erreur claire et cohérente.
"""

from typing import Optional, Dict, Any, List


class EnrichmentServiceError(Exception):
    """Exception de base pour le service d'enrichissement."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or "ENRICHMENT_ERROR"
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Code: {self.error_code}, Details: {self.details})"
        return f"{self.message} (Code: {self.error_code})"


class EmbeddingServiceError(EnrichmentServiceError):
    """Erreurs liées au service d'embedding."""
    
    def __init__(self, message: str, model: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            details={**(details or {}), "model": model}
        )


class QdrantServiceError(EnrichmentServiceError):
    """Erreurs liées au service Qdrant."""
    
    def __init__(self, message: str, collection: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="QDRANT_ERROR",
            details={**(details or {}), "collection": collection}
        )


class TransactionEnrichmentError(EnrichmentServiceError):
    """Erreurs liées à l'enrichissement des transactions."""
    
    def __init__(self, message: str, transaction_id: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TRANSACTION_ENRICHMENT_ERROR",
            details={**(details or {}), "transaction_id": transaction_id}
        )


class PatternDetectionError(EnrichmentServiceError):
    """Erreurs liées à la détection de patterns."""
    
    def __init__(self, message: str, pattern_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PATTERN_DETECTION_ERROR",
            details={**(details or {}), "pattern_type": pattern_type}
        )


class InsightGenerationError(EnrichmentServiceError):
    """Erreurs liées à la génération d'insights."""
    
    def __init__(self, message: str, insight_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="INSIGHT_GENERATION_ERROR",
            details={**(details or {}), "insight_type": insight_type}
        )


class SummaryGenerationError(EnrichmentServiceError):
    """Erreurs liées à la génération de résumés."""
    
    def __init__(self, message: str, period_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SUMMARY_GENERATION_ERROR",
            details={**(details or {}), "period_type": period_type}
        )


class AccountProfilingError(EnrichmentServiceError):
    """Erreurs liées au profilage des comptes."""
    
    def __init__(self, message: str, account_id: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="ACCOUNT_PROFILING_ERROR",
            details={**(details or {}), "account_id": account_id}
        )


class ValidationError(EnrichmentServiceError):
    """Erreurs de validation des données."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field, "value": str(value) if value is not None else None}
        )


class ConfigurationError(EnrichmentServiceError):
    """Erreurs de configuration du service."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key}
        )


class BatchProcessingError(EnrichmentServiceError):
    """Erreurs liées au traitement par lots."""
    
    def __init__(self, message: str, batch_id: Optional[str] = None, failed_items: Optional[List[str]] = None):
        super().__init__(
            message=message,
            error_code="BATCH_PROCESSING_ERROR",
            details={
                "batch_id": batch_id,
                "failed_items": failed_items or [],
                "failed_count": len(failed_items) if failed_items else 0
            }
        )


class TriggerProcessingError(EnrichmentServiceError):
    """Erreurs liées au traitement des triggers PostgreSQL."""
    
    def __init__(self, message: str, trigger_event: Optional[str] = None, payload: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TRIGGER_PROCESSING_ERROR",
            details={
                "trigger_event": trigger_event,
                "payload": payload
            }
        )


class RateLimitExceededError(EnrichmentServiceError):
    """Erreur de dépassement de limite de taux."""
    
    def __init__(self, message: str, user_id: Optional[int] = None, limit: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "user_id": user_id,
                "limit": limit
            }
        )


class InsufficientDataError(EnrichmentServiceError):
    """Erreur lorsque les données sont insuffisantes pour l'enrichissement."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, minimum_required: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA",
            details={
                "data_type": data_type,
                "minimum_required": minimum_required
            }
        )


class ExternalServiceError(EnrichmentServiceError):
    """Erreurs liées aux services externes."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={
                "service_name": service_name,
                "status_code": status_code
            }
        )


# Fonctions utilitaires pour la gestion d'erreurs

def handle_embedding_error(error: Exception, context: Dict[str, Any]) -> EmbeddingServiceError:
    """
    Convertit une erreur générique en EmbeddingServiceError avec contexte.
    
    Args:
        error: Exception originale
        context: Contexte de l'erreur
        
    Returns:
        EmbeddingServiceError: Exception enrichie
    """
    return EmbeddingServiceError(
        message=f"Embedding generation failed: {str(error)}",
        model=context.get("model"),
        details={
            "original_error": str(error),
            "error_type": type(error).__name__,
            **context
        }
    )


def handle_qdrant_error(error: Exception, context: Dict[str, Any]) -> QdrantServiceError:
    """
    Convertit une erreur générique en QdrantServiceError avec contexte.
    
    Args:
        error: Exception originale
        context: Contexte de l'erreur
        
    Returns:
        QdrantServiceError: Exception enrichie
    """
    return QdrantServiceError(
        message=f"Qdrant operation failed: {str(error)}",
        collection=context.get("collection"),
        details={
            "original_error": str(error),
            "error_type": type(error).__name__,
            "operation": context.get("operation"),
            **context
        }
    )


def handle_enrichment_error(error: Exception, enrichment_type: str, context: Dict[str, Any]) -> EnrichmentServiceError:
    """
    Convertit une erreur générique en exception d'enrichissement appropriée.
    
    Args:
        error: Exception originale
        enrichment_type: Type d'enrichissement
        context: Contexte de l'erreur
        
    Returns:
        EnrichmentServiceError: Exception appropriée selon le type
    """
    error_classes = {
        "transaction": TransactionEnrichmentError,
        "pattern": PatternDetectionError,
        "insight": InsightGenerationError,
        "summary": SummaryGenerationError,
        "account": AccountProfilingError
    }
    
    error_class = error_classes.get(enrichment_type, EnrichmentServiceError)
    
    return error_class(
        message=f"{enrichment_type.title()} enrichment failed: {str(error)}",
        details={
            "original_error": str(error),
            "error_type": type(error).__name__,
            **context
        }
    )


def format_error_response(error: EnrichmentServiceError) -> Dict[str, Any]:
    """
    Formate une exception d'enrichissement pour une réponse API.
    
    Args:
        error: Exception d'enrichissement
        
    Returns:
        Dict: Réponse formatée pour l'API
    """
    return {
        "error": {
            "code": error.error_code,
            "message": error.message,
            "details": error.details
        }
    }


def is_retriable_error(error: Exception) -> bool:
    """
    Détermine si une erreur peut être retentée.
    
    Args:
        error: Exception à vérifier
        
    Returns:
        bool: True si l'erreur peut être retentée
    """
    # Erreurs de configuration et de validation ne sont pas retriables
    non_retriable_errors = (
        ConfigurationError,
        ValidationError,
        RateLimitExceededError,
        InsufficientDataError
    )
    
    if isinstance(error, non_retriable_errors):
        return False
    
    # Erreurs de services externes peuvent être retentées
    if isinstance(error, (EmbeddingServiceError, QdrantServiceError, ExternalServiceError)):
        return True
    
    # Autres erreurs d'enrichissement peuvent être retentées
    if isinstance(error, EnrichmentServiceError):
        return True
    
    return False


def get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """
    Calcule le délai avant une nouvelle tentative avec backoff exponentiel.
    
    Args:
        attempt: Numéro de tentative (commence à 1)
        base_delay: Délai de base en secondes
        max_delay: Délai maximum en secondes
        
    Returns:
        float: Délai en secondes
    """
    import random
    
    # Backoff exponentiel avec jitter
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    
    # Ajouter un jitter aléatoire de ±25%
    jitter = delay * 0.25 * (2 * random.random() - 1)
    
    return max(0.1, delay + jitter)