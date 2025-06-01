"""
Modèles de données pour le service de conversation.

Ce module définit les structures de données utilisées pour les conversations,
les intentions détectées et les réponses générées.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
from enum import Enum
import uuid


class IntentType(str, Enum):
    """Types d'intention détectés."""
    SEARCH_TRANSACTIONS = "search_transactions"
    ACCOUNT_SUMMARY = "account_summary"
    SPENDING_ANALYSIS = "spending_analysis"
    BUDGET_INQUIRY = "budget_inquiry"
    CATEGORY_ANALYSIS = "category_analysis"
    MERCHANT_ANALYSIS = "merchant_analysis"
    TIME_ANALYSIS = "time_analysis"
    COMPARISON = "comparison"
    GENERAL_QUESTION = "general_question"
    CLARIFICATION_NEEDED = "clarification_needed"
    GREETING = "greeting"
    HELP = "help"
    UNKNOWN = "unknown"


class MessageRole(str, Enum):
    """Rôles des messages dans une conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationStatus(str, Enum):
    """États d'une conversation."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


# Modèles de base pour les messages
class Message(BaseModel):
    """Message dans une conversation."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = Field(..., description="Rôle du message")
    content: str = Field(..., description="Contenu du message")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées supplémentaires")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg-123",
                "role": "user",
                "content": "Combien j'ai dépensé au restaurant ce mois-ci ?",
                "timestamp": "2024-01-15T10:30:00",
                "metadata": {"intent": "spending_analysis"}
            }
        }


# Modèles pour la détection d'intention
class DetectedIntent(BaseModel):
    """Intention détectée dans un message utilisateur."""
    intent_type: IntentType = Field(..., description="Type d'intention")
    confidence: float = Field(..., ge=0, le=1, description="Niveau de confiance")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres extraits")
    reasoning: Optional[str] = Field(default=None, description="Explication du raisonnement")
    
    class Config:
        json_schema_extra = {
            "example": {
                "intent_type": "search_transactions",
                "confidence": 0.95,
                "parameters": {
                    "merchant_type": "restaurant",
                    "time_period": "ce mois-ci",
                    "amount_type": "total"
                },
                "reasoning": "L'utilisateur demande un total des dépenses dans une catégorie spécifique"
            }
        }


# Modèles pour les requêtes de conversation
class ConversationRequest(BaseModel):
    """Requête pour démarrer ou continuer une conversation."""
    user_id: int = Field(..., description="ID de l'utilisateur")
    message: str = Field(..., min_length=1, description="Message de l'utilisateur")
    conversation_id: Optional[str] = Field(default=None, description="ID de conversation existante")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Contexte additionnel")
    stream: bool = Field(default=True, description="Utiliser le streaming")
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "message": "Combien j'ai dépensé en courses ce mois ?",
                "conversation_id": "conv-456",
                "context": {"timezone": "Europe/Paris"},
                "stream": True
            }
        }


# Modèles pour les réponses
class StreamChunk(BaseModel):
    """Chunk de réponse en streaming."""
    type: Literal["content", "intent", "search_results", "error", "done"] = Field(..., description="Type de chunk")
    content: Optional[str] = Field(default=None, description="Contenu textuel")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Données structurées")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "content",
                "content": "Voici vos dépenses en courses ce mois :",
                "data": None,
                "metadata": {"token_count": 8}
            }
        }


class ConversationResponse(BaseModel):
    """Réponse complète de conversation."""
    conversation_id: str = Field(..., description="ID de la conversation")
    message_id: str = Field(..., description="ID du message de réponse")
    content: str = Field(..., description="Contenu de la réponse")
    intent: DetectedIntent = Field(..., description="Intention détectée")
    search_results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Résultats de recherche")
    processing_time: float = Field(..., description="Temps de traitement")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Utilisation des tokens")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées")
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "conv-456",
                "message_id": "msg-789",
                "content": "Ce mois-ci, vous avez dépensé 245,67€ en courses...",
                "intent": {
                    "intent_type": "spending_analysis",
                    "confidence": 0.95,
                    "parameters": {"category": "courses", "period": "ce mois"}
                },
                "search_results": [{"amount": 245.67, "count": 12}],
                "processing_time": 1.25,
                "token_usage": {"input": 50, "output": 100}
            }
        }


# Modèles pour la gestion des conversations
class ConversationSummary(BaseModel):
    """Résumé d'une conversation."""
    id: str = Field(..., description="ID de la conversation")
    user_id: int = Field(..., description="ID de l'utilisateur")
    title: Optional[str] = Field(default=None, description="Titre de la conversation")
    last_message: str = Field(..., description="Dernier message")
    message_count: int = Field(..., description="Nombre de messages")
    created_at: datetime = Field(..., description="Date de création")
    updated_at: datetime = Field(..., description="Dernière mise à jour")
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "conv-456",
                "user_id": 123,
                "title": "Analyse des dépenses",
                "last_message": "Merci pour ces informations !",
                "message_count": 8,
                "created_at": "2024-01-15T10:00:00",
                "updated_at": "2024-01-15T10:30:00",
                "status": "active"
            }
        }


class ConversationDetail(BaseModel):
    """Détail complet d'une conversation."""
    id: str = Field(..., description="ID de la conversation")
    user_id: int = Field(..., description="ID de l'utilisateur")
    title: Optional[str] = Field(default=None, description="Titre de la conversation")
    messages: List[Message] = Field(..., description="Liste des messages")
    created_at: datetime = Field(..., description="Date de création")
    updated_at: datetime = Field(..., description="Dernière mise à jour")
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "conv-456",
                "user_id": 123,
                "title": "Analyse des dépenses",
                "messages": [
                    {
                        "id": "msg-1",
                        "role": "user",
                        "content": "Bonjour",
                        "timestamp": "2024-01-15T10:00:00"
                    }
                ],
                "created_at": "2024-01-15T10:00:00",
                "updated_at": "2024-01-15T10:30:00",
                "status": "active"
            }
        }


# Modèles pour la configuration
class ConversationConfig(BaseModel):
    """Configuration d'une conversation."""
    max_history_length: int = Field(default=20, description="Nombre max de messages dans l'historique")
    context_window: int = Field(default=10, description="Fenêtre de contexte pour les messages")
    enable_search: bool = Field(default=True, description="Activer la recherche automatique")
    enable_intent_detection: bool = Field(default=True, description="Activer la détection d'intention")
    response_style: str = Field(default="helpful", description="Style de réponse")
    language: str = Field(default="fr", description="Langue préférée")
    
    class Config:
        json_schema_extra = {
            "example": {
                "max_history_length": 20,
                "context_window": 10,
                "enable_search": True,
                "enable_intent_detection": True,
                "response_style": "helpful",
                "language": "fr"
            }
        }


# Modèles pour les statistiques
class ConversationStats(BaseModel):
    """Statistiques d'utilisation des conversations."""
    user_id: int = Field(..., description="ID de l'utilisateur")
    total_conversations: int = Field(..., description="Nombre total de conversations")
    total_messages: int = Field(..., description="Nombre total de messages")
    avg_messages_per_conversation: float = Field(..., description="Moyenne de messages par conversation")
    most_common_intents: List[Dict[str, Union[str, int]]] = Field(..., description="Intentions les plus fréquentes")
    total_tokens_used: Optional[int] = Field(default=None, description="Total de tokens utilisés")
    last_conversation_date: Optional[datetime] = Field(default=None, description="Date de dernière conversation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "total_conversations": 25,
                "total_messages": 180,
                "avg_messages_per_conversation": 7.2,
                "most_common_intents": [
                    {"intent": "search_transactions", "count": 45},
                    {"intent": "spending_analysis", "count": 32}
                ],
                "total_tokens_used": 15420,
                "last_conversation_date": "2024-01-15T15:30:00"
            }
        }


# Modèle pour les erreurs
class ConversationError(BaseModel):
    """Erreur dans le traitement d'une conversation."""
    error_code: str = Field(..., description="Code d'erreur")
    message: str = Field(..., description="Message d'erreur")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Détails de l'erreur")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "INTENT_DETECTION_FAILED",
                "message": "Impossible de détecter l'intention du message",
                "details": {"raw_message": "...", "confidence_threshold": 0.5},
                "timestamp": "2024-01-15T10:30:00"
            }
        }