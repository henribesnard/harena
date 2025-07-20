"""
Modèles Pydantic pour les endpoints de conversation
Intégration avec système de détection d'intention
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ConversationContext(BaseModel):
    """Contexte conversationnel enrichi"""
    previous_messages: Optional[List[str]] = Field(default_factory=list, max_items=10)
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = None
    channel: Optional[str] = Field(default="web", description="Canal de communication")
    
    @validator('previous_messages')
    def validate_messages(cls, v):
        if v and len(v) > 10:
            return v[-10:]  # Garde seulement les 10 derniers messages
        return v


class ConversationRequest(BaseModel):
    """Requête de conversation avec contexte"""
    message: str = Field(..., min_length=1, max_length=2000, description="Message utilisateur")
    context: ConversationContext = Field(default_factory=ConversationContext)
    user_id: Optional[int] = Field(None, description="ID utilisateur pour personnalisation")
    
    @validator('message')
    def validate_message(cls, v):
        # Nettoyage basique message
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Quel est le solde de mon compte courant ?",
                "context": {
                    "previous_messages": ["Bonjour", "Je voudrais consulter mes comptes"],
                    "user_preferences": {"language": "fr", "currency": "EUR"},
                    "channel": "web"
                },
                "user_id": 12345
            }
        }


class ConversationMetadata(BaseModel):
    """Métadonnées réponse avec métriques performance"""
    processing_time_ms: int = Field(..., description="Temps traitement en millisecondes")
    intent_level: str = Field(..., description="Niveau détection utilisé (L0/L1/L2)")
    cache_hit: bool = Field(default=False, description="Cache hit pour optimisation")
    model_used: str = Field(..., description="Modèle/service utilisé")
    entities_extracted: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "processing_time_ms": 45,
                "intent_level": "L0_pattern",
                "cache_hit": True,
                "model_used": "pattern_matcher",
                "entities_extracted": {"account_type": "compte_courant"},
                "timestamp": "2025-01-20T10:30:00Z"
            }
        }


class ActionSuggestion(BaseModel):
    """Suggestion d'action utilisateur"""
    text: str = Field(..., description="Texte suggestion")
    action_type: str = Field(..., description="Type d'action suggérée")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Voir l'historique des transactions",
                "action_type": "transaction_history",
                "parameters": {"account_id": "12345", "limit": 20},
                "confidence": 0.85
            }
        }


class ConversationResponse(BaseModel):
    """Réponse conversation complète"""
    response: str = Field(..., description="Réponse générée")
    conversation_id: str = Field(..., description="ID unique conversation")
    intent_detected: str = Field(..., description="Intention détectée")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Score confiance intention")
    suggestions: List[ActionSuggestion] = Field(default_factory=list)
    metadata: ConversationMetadata = Field(..., description="Métadonnées performance")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Votre compte courant affiche un solde de 2,847.32€.",
                "conversation_id": "conv_123456789",
                "intent_detected": "balance_check",
                "confidence_score": 0.95,
                "suggestions": [
                    {
                        "text": "Voir l'historique des transactions",
                        "action_type": "transaction_history",
                        "parameters": {"account_id": "12345"},
                        "confidence": 0.85
                    }
                ],
                "metadata": {
                    "processing_time_ms": 45,
                    "intent_level": "L0_pattern",
                    "cache_hit": True,
                    "model_used": "pattern_matcher",
                    "entities_extracted": {"account_type": "compte_courant"}
                }
            }
        }
