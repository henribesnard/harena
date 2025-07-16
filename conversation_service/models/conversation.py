# conversation_service/models/conversation.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid

class FinancialIntent(str, Enum):
    """Énumération des intentions financières supportées"""
    SEARCH_BY_MERCHANT = "search_by_merchant"      # "mes achats netflix"
    SEARCH_BY_CATEGORY = "search_by_category"      # "mes restaurants"
    SEARCH_BY_AMOUNT = "search_by_amount"          # "plus de 100€"
    SEARCH_BY_DATE = "search_by_date"              # "janvier 2024"
    SEARCH_GENERAL = "search_general"              # "mes transactions"
    SPENDING_ANALYSIS = "spending_analysis"        # "combien dépensé"
    INCOME_ANALYSIS = "income_analysis"            # "mes revenus"
    UNCLEAR_INTENT = "unclear_intent"              # Cas ambigus

class ConversationContext(BaseModel):
    """Contexte de la conversation"""
    session_id: str = Field(..., description="ID de session unique")
    previous_messages: List[str] = Field(default_factory=list, description="Messages précédents")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="Préférences utilisateur")
    
    model_config = {"extra": "allow"}

class ChatRequest(BaseModel):
    """Requête de conversation"""
    user_id: int = Field(..., description="ID utilisateur", gt=0)
    message: str = Field(..., description="Message utilisateur", min_length=1, max_length=1000)
    context: Optional[ConversationContext] = Field(default=None, description="Contexte conversation")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Le message ne peut pas être vide")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": 34,
                "message": "mes restaurants ce mois",
                "context": {
                    "session_id": "sess_abc123",
                    "previous_messages": []
                }
            }
        }
    }

class EntityHints(BaseModel):
    """Entités détectées dans le message"""
    merchant: Optional[str] = Field(None, description="Nom du marchand")
    category: Optional[str] = Field(None, description="Catégorie financière")
    amount: Optional[str] = Field(None, description="Montant")
    operator: Optional[str] = Field(None, description="Opérateur de comparaison")
    period: Optional[str] = Field(None, description="Période temporelle")
    date: Optional[str] = Field(None, description="Date spécifique")
    
    model_config = {"extra": "allow"}

class IntentResult(BaseModel):
    """Résultat de classification d'intention"""
    intent: FinancialIntent = Field(..., description="Intention classifiée")
    confidence: float = Field(..., description="Niveau de confiance", ge=0.0, le=1.0)
    entities: EntityHints = Field(default_factory=EntityHints, description="Entités détectées")
    reasoning: Optional[str] = Field(None, description="Raisonnement de la classification")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("La confiance doit être entre 0.0 et 1.0")
        return round(v, 3)

class ProcessingMetadata(BaseModel):
    """Métadonnées de traitement"""
    processing_time_ms: int = Field(..., description="Temps de traitement en millisecondes")
    agent_used: str = Field(..., description="Agent utilisé")
    model_used: str = Field(..., description="Modèle utilisé")
    tokens_consumed: Optional[int] = Field(None, description="Tokens consommés")
    cache_hit: bool = Field(default=False, description="Cache utilisé")
    
    model_config = {"extra": "allow"}

class ChatResponse(BaseModel):
    """Réponse de conversation"""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID conversation")
    response: str = Field(..., description="Réponse textuelle")
    intent: FinancialIntent = Field(..., description="Intention détectée")
    confidence: float = Field(..., description="Niveau de confiance")
    entities: EntityHints = Field(default_factory=EntityHints, description="Entités détectées")
    is_clear: bool = Field(..., description="Intention claire")
    clarification_needed: Optional[str] = Field(None, description="Clarification nécessaire")
    metadata: ProcessingMetadata = Field(..., description="Métadonnées de traitement")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "conversation_id": "conv_xyz789",
                "response": "Intention détectée : Recherche par catégorie",
                "intent": "search_by_category",
                "confidence": 0.90,
                "entities": {
                    "category": "restaurant",
                    "period": "ce mois"
                },
                "is_clear": True,
                "clarification_needed": None,
                "metadata": {
                    "processing_time_ms": 245,
                    "agent_used": "intent_classifier",
                    "model_used": "deepseek-chat",
                    "cache_hit": False
                }
            }
        }
    }

class HealthResponse(BaseModel):
    """Réponse du endpoint de santé"""
    status: str = Field(..., description="État du service")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Version du service")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="État des dépendances")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "dependencies": {
                    "deepseek": "connected",
                    "cache": "operational"
                }
            }
        }
    }

class MetricsResponse(BaseModel):
    """Réponse des métriques"""
    total_classifications: int = Field(..., description="Nombre total de classifications")
    success_rate: float = Field(..., description="Taux de succès")
    avg_processing_time_ms: float = Field(..., description="Temps moyen de traitement")
    avg_confidence: float = Field(..., description="Confiance moyenne")
    intent_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution des intentions")
    cache_hit_rate: float = Field(..., description="Taux de cache hit")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_classifications": 1247,
                "success_rate": 0.95,
                "avg_processing_time_ms": 245.7,
                "avg_confidence": 0.87,
                "intent_distribution": {
                    "search_by_merchant": 450,
                    "search_by_category": 320,
                    "search_by_amount": 200
                },
                "cache_hit_rate": 0.35
            }
        }
    }

class ConfigResponse(BaseModel):
    """Réponse de configuration"""
    service_name: str = Field(..., description="Nom du service")
    version: str = Field(..., description="Version")
    configuration: Dict[str, Any] = Field(..., description="Configuration publique")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "service_name": "Conversation Service",
                "version": "1.0.0",
                "configuration": {
                    "min_confidence_threshold": 0.7,
                    "supported_intents": ["search_by_merchant", "search_by_category"],
                    "cache_enabled": True
                }
            }
        }
    }

# Types d'erreur personnalisés
class ConversationError(BaseModel):
    """Erreur de conversation"""
    error_type: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur")
    details: Optional[Dict[str, Any]] = Field(None, description="Détails de l'erreur")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ValidationError(ConversationError):
    """Erreur de validation"""
    error_type: str = Field(default="validation_error")

class ProcessingError(ConversationError):
    """Erreur de traitement"""
    error_type: str = Field(default="processing_error")

class DeepSeekError(ConversationError):
    """Erreur DeepSeek"""
    error_type: str = Field(default="deepseek_error")