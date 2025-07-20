"""
Modèles Pydantic pour le Conversation Service
Architecture hybride avec détection d'intention
"""

from .conversation_models import (
    ConversationRequest,
    ConversationResponse,
    ConversationContext,
    ConversationMetadata,
    ActionSuggestion
)
from .intent_models import (
    IntentDetectionRequest,
    IntentDetectionResponse,
    IntentEntity,
    IntentPattern,
    IntentEmbedding
)
from .service_contracts import (
    SearchServiceQuery,
    SearchServiceResponse,
    SearchServiceMetadata,
    SearchFilter,
    SearchAggregation
)

__version__ = "1.0.0"
__all__ = [
    # Conversation models
    "ConversationRequest",
    "ConversationResponse", 
    "ConversationContext",
    "ConversationMetadata",
    "ActionSuggestion",
    
    # Intent models
    "IntentDetectionRequest",
    "IntentDetectionResponse",
    "IntentEntity",
    "IntentPattern",
    "IntentEmbedding",
    
    # Service contracts
    "SearchServiceQuery",
    "SearchServiceResponse",
    "SearchServiceMetadata",
    "SearchFilter",
    "SearchAggregation"
]