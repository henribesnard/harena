"""
Modèles Pydantic V2 pour les réponses conversation
"""
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from conversation_service.prompts.harena_intents import HarenaIntentType

class IntentAlternative(BaseModel):
    """Alternative d'intention avec score"""
    intent_type: HarenaIntentType
    confidence: float
    reasoning: Optional[str] = None
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence doit être entre 0.0 et 1.0")
        return v

class IntentClassificationResult(BaseModel):
    """Résultat de classification d'intention"""
    intent_type: HarenaIntentType
    confidence: float
    reasoning: str
    original_message: str
    category: str
    is_supported: bool
    alternatives: List[IntentAlternative] = []
    processing_time_ms: Optional[int] = None
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence doit être entre 0.0 et 1.0")
        return v

class AgentMetrics(BaseModel):
    """Métriques d'exécution des agents"""
    agent_used: str
    cache_hit: bool
    model_used: str
    tokens_consumed: int
    confidence_threshold_met: bool

class ConversationResponse(BaseModel):
    """Réponse complète Phase 1 - Intentions uniquement"""
    user_id: int
    message: str
    timestamp: datetime
    processing_time_ms: int
    intent: IntentClassificationResult
    agent_metrics: AgentMetrics
    
    @field_validator('processing_time_ms')
    @classmethod
    def validate_processing_time(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Processing time ne peut pas être négatif")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123,
                "message": "Mes achats Amazon",
                "timestamp": "2024-08-26T14:30:00+00:00",
                "processing_time_ms": 245,
                "intent": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "confidence": 0.94,
                    "reasoning": "L'utilisateur cherche ses transactions Amazon",
                    "original_message": "Mes achats Amazon",
                    "category": "FINANCIAL_QUERY",
                    "is_supported": True,
                    "alternatives": [],
                    "processing_time_ms": 245
                },
                "agent_metrics": {
                    "agent_used": "intent_classifier",
                    "cache_hit": False,
                    "model_used": "deepseek-chat",
                    "tokens_consumed": 156,
                    "confidence_threshold_met": True
                }
            }
        }