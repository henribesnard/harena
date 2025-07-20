"""
Modèles spécifiques à la détection d'intention
Support architecture hybride L0/L1/L2
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class IntentLevel(str, Enum):
    """Niveaux de détection d'intention"""
    L0_PATTERN = "L0_pattern"
    L1_LIGHTWEIGHT = "L1_lightweight"
    L2_LLM = "L2_llm"
    FALLBACK = "fallback"


class IntentType(str, Enum):
    """Types d'intentions financières"""
    BALANCE_CHECK = "balance_check"
    EXPENSE_ANALYSIS = "expense_analysis"
    TRANSFER = "transfer" 
    TRANSACTION_SEARCH = "transaction_search"
    BUDGET_INQUIRY = "budget_inquiry"
    ACCOUNT_MANAGEMENT = "account_management"
    GENERAL_QUERY = "general_query"


class IntentEntity(BaseModel):
    """Entité extraite avec métadonnées"""
    name: str = Field(..., description="Nom de l'entité")
    value: str = Field(..., description="Valeur extraite")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance extraction")
    type: str = Field(..., description="Type d'entité (amount, date, category, etc.)")
    span: Optional[Dict[str, int]] = Field(None, description="Position dans le texte")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "amount",
                "value": "50.00",
                "confidence": 0.92,
                "type": "monetary",
                "span": {"start": 15, "end": 20}
            }
        }


class IntentPattern(BaseModel):
    """Pattern financier pré-compilé"""
    pattern_id: str = Field(..., description="ID unique du pattern")
    pattern_text: str = Field(..., description="Texte du pattern")
    intent_type: IntentType = Field(..., description="Intention associée")
    entities: List[str] = Field(default_factory=list, description="Entités à extraire")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance pattern")
    examples: List[str] = Field(default_factory=list, description="Exemples d'usage")
    
    class Config:
        schema_extra = {
            "example": {
                "pattern_id": "balance_check_001",
                "pattern_text": "solde compte",
                "intent_type": "balance_check",
                "entities": ["account_type"],
                "confidence": 0.95,
                "examples": ["solde compte courant", "combien sur mon compte"]
            }
        }


class IntentEmbedding(BaseModel):
    """Embedding d'intention TinyBERT"""
    intent_type: IntentType = Field(..., description="Type d'intention")
    embedding: List[float] = Field(..., description="Vecteur embedding 384-dim")
    keywords: List[str] = Field(default_factory=list, description="Mots-clés associés")
    confidence_threshold: float = Field(default=0.85, description="Seuil de confiance")
    
    @validator('embedding')
    def validate_embedding_size(cls, v):
        if len(v) != 384:
            raise ValueError('Embedding must be 384-dimensional for TinyBERT')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "intent_type": "balance_check",
                "embedding": [0.123, 0.456],  # Truncated for example
                "keywords": ["solde", "combien", "montant"],
                "confidence_threshold": 0.85
            }
        }


class IntentDetectionRequest(BaseModel):
    """Requête détection d'intention"""
    query: str = Field(..., min_length=1, max_length=2000, description="Requête utilisateur")
    user_id: int = Field(..., description="ID utilisateur")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contexte conversationnel")
    force_level: Optional[IntentLevel] = Field(None, description="Forcer un niveau spécifique")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Quel est le solde de mon compte courant ?",
                "user_id": 12345,
                "context": {"previous_intent": "greeting"},
                "force_level": None
            }
        }


class IntentDetectionResponse(BaseModel):
    """Réponse détection d'intention"""
    intent_type: IntentType = Field(..., description="Intention détectée")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score de confiance")
    level_used: IntentLevel = Field(..., description="Niveau de détection utilisé")
    entities: List[IntentEntity] = Field(default_factory=list, description="Entités extraites")
    processing_time_ms: int = Field(..., description="Temps de traitement")
    cache_hit: bool = Field(default=False, description="Hit cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées additionnelles")
    
    class Config:
        schema_extra = {
            "example": {
                "intent_type": "balance_check",
                "confidence": 0.95,
                "level_used": "L0_pattern",
                "entities": [
                    {
                        "name": "account_type",
                        "value": "compte_courant",
                        "confidence": 0.92,
                        "type": "account",
                        "span": {"start": 25, "end": 39}
                    }
                ],
                "processing_time_ms": 12,
                "cache_hit": True,
                "metadata": {"pattern_matched": "balance_check_001"}
            }
        }
