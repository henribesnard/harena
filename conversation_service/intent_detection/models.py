"""
Modèles pour le système de détection d'intention
"""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime


class IntentLevel(Enum):
    """Niveau de détection d'intention utilisé"""
    L0_PATTERN = "l0_pattern"        # Cache patterns pré-calculés
    L1_LIGHTWEIGHT = "l1_lightweight" # TinyBERT + cache
    L2_LLM = "l2_llm"               # DeepSeek API fallback
    FALLBACK = "fallback"           # Fallback d'urgence


class IntentType(Enum):
    """Types d'intentions financières supportées"""
    BALANCE_CHECK = "balance_check"
    EXPENSE_ANALYSIS = "expense_analysis" 
    TRANSFER = "transfer"
    TRANSACTION_SEARCH = "transaction_search"
    BUDGET_INQUIRY = "budget_inquiry"
    ACCOUNT_MANAGEMENT = "account_management"
    GENERAL_QUERY = "general_query"


@dataclass
class IntentConfidence:
    """Score de confiance avec métadonnées"""
    score: float                    # 0.0 à 1.0
    level: IntentLevel             # Niveau de détection utilisé
    threshold_used: float = 0.85   # Seuil utilisé pour validation
    
    @property
    def is_confident(self) -> bool:
        return self.score >= self.threshold_used
    
    @property
    def confidence_category(self) -> str:
        if self.score >= 0.95:
            return "very_high"
        elif self.score >= 0.85:
            return "high" 
        elif self.score >= 0.70:
            return "medium"
        else:
            return "low"


@dataclass
class IntentResult:
    """Résultat complet de détection d'intention"""
    intent_type: str               # Type d'intention détectée
    entities: Dict[str, Any]       # Entités extraites (montants, dates, etc.)
    confidence: IntentConfidence   # Score et métadonnées confiance
    level: IntentLevel            # Niveau de détection utilisé
    latency_ms: int               # Temps de traitement en ms
    metadata: Dict[str, Any]      # Métadonnées additionnelles
    
    timestamp: datetime = None    # Timestamp détection
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @classmethod
    def from_cache(
        cls, 
        cached_data: Dict[str, Any], 
        level: IntentLevel,
        cache_hit: bool = True
    ) -> 'IntentResult':
        """Reconstruction depuis cache Redis"""
        return cls(
            intent_type=cached_data["intent_type"],
            entities=cached_data.get("entities", {}),
            confidence=IntentConfidence(
                score=cached_data["confidence_score"],
                level=level,
                threshold_used=cached_data.get("threshold_used", 0.85)
            ),
            level=level,
            latency_ms=cached_data.get("latency_ms", 0),
            metadata={
                **cached_data.get("metadata", {}),
                "cache_hit": cache_hit,
                "cached_at": cached_data.get("cached_at")
            }
        )
    
    def to_cache(self) -> Dict[str, Any]:
        """Sérialisation pour cache Redis"""
        return {
            "intent_type": self.intent_type,
            "entities": self.entities,
            "confidence_score": self.confidence.score,
            "threshold_used": self.confidence.threshold_used,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "cached_at": self.timestamp.isoformat(),
            "level": self.level.value
        }
    
    @property
    def is_actionable(self) -> bool:
        """Détermine si l'intention est exploitable"""
        return (
            self.confidence.is_confident and 
            self.intent_type != "general_query" and
            len(self.entities) > 0
        )
    
    def get_search_parameters(self) -> Dict[str, Any]:
        """Conversion en paramètres de recherche pour Search Service"""
        search_params = {
            "intent_type": self.intent_type,
            "confidence_score": self.confidence.score,
            "detection_level": self.level.value
        }
        
        # Extraction entités pour filtres de recherche
        if "category" in self.entities:
            search_params["category_filter"] = self.entities["category"]
        
        if "date_range" in self.entities:
            search_params["date_filter"] = self.entities["date_range"]
            
        if "amount_range" in self.entities:
            search_params["amount_filter"] = self.entities["amount_range"]
            
        return search_params