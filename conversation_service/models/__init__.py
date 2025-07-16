# conversation_service/models/__init__.py
"""
Models module pour le Conversation Service

Ce module contient tous les modèles Pydantic utilisés par le service :
- Modèles de requête/réponse API
- Énumérations des intentions financières
- Modèles d'erreur
- Modèles de métriques et configuration
"""

from .conversation import (
    # Énumérations
    FinancialIntent,
    
    # Modèles de données core
    ConversationContext,
    EntityHints,
    IntentResult,
    
    # Modèles API
    ChatRequest,
    ChatResponse,
    ProcessingMetadata,
    
    # Modèles système
    HealthResponse,
    MetricsResponse,
    ConfigResponse,
    
    # Modèles d'erreur
    ConversationError,
    ValidationError,
    ProcessingError,
    DeepSeekError
)

__version__ = "1.0.0"
__all__ = [
    # Énumérations
    "FinancialIntent",
    
    # Modèles de données
    "ConversationContext",
    "EntityHints", 
    "IntentResult",
    
    # Modèles API
    "ChatRequest",
    "ChatResponse",
    "ProcessingMetadata",
    
    # Modèles système
    "HealthResponse",
    "MetricsResponse",
    "ConfigResponse",
    
    # Modèles d'erreur
    "ConversationError",
    "ValidationError",
    "ProcessingError", 
    "DeepSeekError"
]

# Helpers pour validation
def validate_intent(intent_str: str) -> bool:
    """Valide qu'une chaîne correspond à une intention connue"""
    try:
        FinancialIntent(intent_str)
        return True
    except ValueError:
        return False

def get_supported_intents() -> list[str]:
    """Retourne la liste des intentions supportées"""
    return [intent.value for intent in FinancialIntent]

def create_error_response(error_type: str, message: str, details=None) -> ConversationError:
    """Factory pour créer des réponses d'erreur standardisées"""
    error_classes = {
        "validation_error": ValidationError,
        "processing_error": ProcessingError,
        "deepseek_error": DeepSeekError
    }
    
    error_class = error_classes.get(error_type, ConversationError)
    return error_class(
        error_type=error_type,
        message=message,
        details=details or {}
    )