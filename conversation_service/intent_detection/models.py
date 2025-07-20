"""
📊 Modèles spécialisés détection intentions

Modèles métier pour Intent Detection Engine avec sérialisation Redis,
helpers conversion et gestion des 3 niveaux L0→L1→L2.
"""

import json
import time
from enum import Enum
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, field_validator

# ==========================================
# ENUMS INTENT DETECTION
# ==========================================

class IntentLevel(str, Enum):
    """Niveaux détection Intent Detection Engine"""
    
    L0_PATTERN = "L0_PATTERN"          # Patterns pré-calculés <10ms
    L1_LIGHTWEIGHT = "L1_LIGHTWEIGHT"  # TinyBERT 15-30ms
    L2_LLM = "L2_LLM"                 # DeepSeek 200-500ms
    
    # Niveaux d'erreur
    ERROR_TIMEOUT = "ERROR_TIMEOUT"
    ERROR_FALLBACK = "ERROR_FALLBACK"
    ERROR_VALIDATION = "ERROR_VALIDATION"
    
    @classmethod
    def get_performance_levels(cls) -> List[str]:
        """Retourne les niveaux de performance normaux"""
        return [cls.L0_PATTERN, cls.L1_LIGHTWEIGHT, cls.L2_LLM]
    
    @classmethod
    def get_target_latency(cls, level: "IntentLevel") -> int:
        """Retourne latence cible en ms pour un niveau"""
        targets = {
            cls.L0_PATTERN: 10,
            cls.L1_LIGHTWEIGHT: 30,
            cls.L2_LLM: 500
        }
        return targets.get(level, 1000)

class IntentType(str, Enum):
    """Types d'intentions financières détaillées"""
    
    # ========== INTENTIONS FINANCIÈRES PRINCIPALES ==========
    
    # Consultation de soldes (L0 optimisé)
    BALANCE_CHECK = "BALANCE_CHECK"
    BALANCE_HISTORY = "BALANCE_HISTORY"
    BALANCE_PROJECTION = "BALANCE_PROJECTION"
    
    # Analyse des dépenses (L1/L2)
    EXPENSE_ANALYSIS = "EXPENSE_ANALYSIS"
    EXPENSE_CATEGORY = "EXPENSE_CATEGORY"
    EXPENSE_COMPARISON = "EXPENSE_COMPARISON"
    EXPENSE_TREND = "EXPENSE_TREND"
    
    # Transferts et virements (L0/L1)
    TRANSFER = "TRANSFER"
    TRANSFER_INTERNAL = "TRANSFER_INTERNAL"
    TRANSFER_EXTERNAL = "TRANSFER_EXTERNAL"
    TRANSFER_RECURRING = "TRANSFER_RECURRING"
    
    # Paiements de factures (L1)
    BILL_PAYMENT = "BILL_PAYMENT"
    BILL_SCHEDULE = "BILL_SCHEDULE"
    BILL_HISTORY = "BILL_HISTORY"
    
    # Investissements (L2 complexe)
    INVESTMENT_QUERY = "INVESTMENT_QUERY"
    INVESTMENT_PERFORMANCE = "INVESTMENT_PERFORMANCE"
    INVESTMENT_ALLOCATION = "INVESTMENT_ALLOCATION"
    
    # Prêts et crédits (L1/L2)
    LOAN_INQUIRY = "LOAN_INQUIRY"
    LOAN_APPLICATION = "LOAN_APPLICATION"
    LOAN_STATUS = "LOAN_STATUS"
    
    # Gestion cartes (L0/L1)
    CARD_MANAGEMENT = "CARD_MANAGEMENT"
    CARD_ACTIVATION = "CARD_ACTIVATION"
    CARD_BLOCKING = "CARD_BLOCKING"
    CARD_LIMITS = "CARD_LIMITS"
    
    # Historique transactions (L1)
    TRANSACTION_HISTORY = "TRANSACTION_HISTORY"
    TRANSACTION_SEARCH = "TRANSACTION_SEARCH"
    TRANSACTION_DETAILS = "TRANSACTION_DETAILS"
    
    # Planification budget (L2)
    BUDGET_PLANNING = "BUDGET_PLANNING"
    BUDGET_TRACKING = "BUDGET_TRACKING"
    BUDGET_OPTIMIZATION = "BUDGET_OPTIMIZATION"
    
    # Objectifs épargne (L2)
    SAVINGS_GOAL = "SAVINGS_GOAL"
    SAVINGS_PROGRESS = "SAVINGS_PROGRESS"
    SAVINGS_ADVICE = "SAVINGS_ADVICE"
    
    # Gestion compte (L1)
    ACCOUNT_MANAGEMENT = "ACCOUNT_MANAGEMENT"
    ACCOUNT_OPENING = "ACCOUNT_OPENING"
    ACCOUNT_CLOSING = "ACCOUNT_CLOSING"
    
    # Conseils financiers (L2 avancé)
    FINANCIAL_ADVICE = "FINANCIAL_ADVICE"
    FINANCIAL_PLANNING = "FINANCIAL_PLANNING"
    FINANCIAL_EDUCATION = "FINANCIAL_EDUCATION"
    
    # ========== INTENTIONS SYSTÈME ==========
    UNKNOWN = "UNKNOWN"
    GREETING = "GREETING"
    HELP = "HELP"
    GOODBYE = "GOODBYE"
    
    @classmethod
    def get_l0_optimized_intents(cls) -> List[str]:
        """Intentions optimisées pour reconnaissance L0 patterns"""
        return [
            cls.BALANCE_CHECK, cls.TRANSFER, cls.CARD_ACTIVATION,
            cls.CARD_BLOCKING, cls.GREETING, cls.HELP
        ]
    
    @classmethod
    def get_l2_complex_intents(cls) -> List[str]:
        """Intentions nécessitant L2 LLM pour analyse complexe"""
        return [
            cls.INVESTMENT_QUERY, cls.INVESTMENT_PERFORMANCE,
            cls.BUDGET_PLANNING, cls.BUDGET_OPTIMIZATION,
            cls.SAVINGS_ADVICE, cls.FINANCIAL_ADVICE,
            cls.FINANCIAL_PLANNING, cls.FINANCIAL_EDUCATION
        ]
    
    def requires_entities(self) -> bool:
        """Vérifie si l'intention nécessite extraction d'entités"""
        entity_required = [
            self.TRANSFER, self.BILL_PAYMENT, self.EXPENSE_ANALYSIS,
            self.TRANSACTION_SEARCH, self.INVESTMENT_QUERY
        ]
        return self in entity_required

# ==========================================
# MODÈLES CONFIANCE ET SCORING
# ==========================================

class IntentConfidence(BaseModel):
    """Modèle score confiance avec métadonnées"""
    
    score: float = Field(
        ge=0.0, le=1.0,
        description="Score confiance principal (0.0 à 1.0)"
    )
    
    level_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores par niveau de détection"
    )
    
    reasoning: Optional[str] = Field(
        default=None,
        description="Explication scoring (debug)"
    )
    
    factors: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Facteurs contribuant au score"
    )
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validation et arrondi score principal"""
        return round(float(v), 3)
    
    @classmethod
    def from_pattern_match(cls, match_strength: float, pattern_type: str) -> "IntentConfidence":
        """Crée confiance depuis match pattern L0"""
        return cls(
            score=match_strength,
            level_confidence={"L0_PATTERN": match_strength},
            reasoning=f"Pattern match: {pattern_type}",
            factors={"pattern_strength": match_strength}
        )
    
    @classmethod
    def from_embedding_similarity(cls, similarity: float, model_name: str) -> "IntentConfidence":
        """Crée confiance depuis similarité embedding L1"""
        return cls(
            score=similarity,
            level_confidence={"L1_LIGHTWEIGHT": similarity},
            reasoning=f"Embedding similarity: {model_name}",
            factors={"cosine_similarity": similarity}
        )
    
    @classmethod
    def from_llm_classification(cls, llm_score: float, model_name: str, context_factors: Dict[str, float] = None) -> "IntentConfidence":
        """Crée confiance depuis classification LLM L2"""
        factors = context_factors or {}
        factors["llm_score"] = llm_score
        
        return cls(
            score=llm_score,
            level_confidence={"L2_LLM": llm_score},
            reasoning=f"LLM classification: {model_name}",
            factors=factors
        )
    
    def is_high_confidence(self, threshold: float = 0.85) -> bool:
        """Vérifie si confiance dépasse seuil"""
        return self.score >= threshold
    
    def get_confidence_level(self) -> str:
        """Niveau confiance textuel"""
        if self.score >= 0.95:
            return "excellent"
        elif self.score >= 0.85:
            return "high"
        elif self.score >= 0.70:
            return "medium"
        elif self.score >= 0.50:
            return "low"
        else:
            return "very_low"

# ==========================================
# MODÈLE RÉSULTAT PRINCIPAL
# ==========================================

class IntentResult(BaseModel):
    """Résultat complet détection intention"""
    
    # Résultat principal
    intent_type: IntentType = Field(description="Type intention détectée")
    confidence: IntentConfidence = Field(description="Score confiance détaillé")
    
    # Métadonnées détection
    level: IntentLevel = Field(description="Niveau détection utilisé")
    latency_ms: float = Field(description="Latence détection en ms")
    from_cache: bool = Field(default=False, description="Résultat depuis cache")
    
    # Entités extraites
    entities: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Entités financières extraites"
    )
    
    # Contexte et debug
    processing_details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Détails traitement (debug)"
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp détection"
    )
    
    # Métadonnées utilisateur
    user_id: Optional[str] = Field(default=None, description="ID utilisateur")
    session_context: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Contexte session utilisateur"
    )
    
    @field_validator('latency_ms')
    @classmethod
    def validate_latency(cls, v: float) -> float:
        """Validation latence"""
        return round(float(v), 2)
    
    def to_cache_dict(self) -> Dict[str, Any]:
        """Sérialisation optimisée pour cache Redis"""
        return {
            "intent_type": self.intent_type.value,
            "confidence_score": self.confidence.score,
            "level": self.level.value,
            "latency_ms": self.latency_ms,
            "entities": self.entities,
            "timestamp": self.timestamp,
            "user_id": self.user_id
        }
    
    @classmethod
    def from_cache_dict(cls, cache_data: Dict[str, Any]) -> "IntentResult":
        """Désérialisation depuis cache Redis"""
        confidence = IntentConfidence(score=cache_data["confidence_score"])
        
        return cls(
            intent_type=IntentType(cache_data["intent_type"]),
            confidence=confidence,
            level=IntentLevel(cache_data["level"]),
            latency_ms=cache_data["latency_ms"],
            from_cache=True,
            entities=cache_data.get("entities", {}),
            timestamp=cache_data.get("timestamp", time.time()),
            user_id=cache_data.get("user_id")
        )
    
    def to_api_response(self) -> Dict[str, Any]:
        """Conversion vers format API Response"""
        return {
            "intent": self.intent_type.value,
            "entities": self.entities,
            "confidence": self.confidence.score,
            "processing_metadata": {
                "level_used": self.level.value,
                "processing_time_ms": self.latency_ms,
                "cache_hit": self.from_cache,
                "timestamp": int(self.timestamp)
            }
        }
    
    def meets_performance_target(self) -> bool:
        """Vérifie si performance respecte cible niveau"""
        target_latency = IntentLevel.get_target_latency(self.level)
        return self.latency_ms <= target_latency
    
    def get_performance_grade(self) -> str:
        """Grade performance A/B/C/D/F"""
        target = IntentLevel.get_target_latency(self.level)
        ratio = self.latency_ms / target
        
        if ratio <= 0.5:
            return "A"
        elif ratio <= 1.0:
            return "B"
        elif ratio <= 1.5:
            return "C"
        elif ratio <= 2.0:
            return "D"
        else:
            return "F"

# ==========================================
# MODÈLES CACHE ET OPTIMISATION
# ==========================================

class CacheKey(BaseModel):
    """Clé cache structurée avec namespace"""
    
    namespace: str = Field(description="Namespace cache (L0/L1/L2)")
    user_id: Optional[str] = Field(default=None, description="ID utilisateur")
    query_hash: str = Field(description="Hash requête")
    version: str = Field(default="v1", description="Version cache")
    
    def to_redis_key(self) -> str:
        """Génère clé Redis standardisée"""
        parts = ["conversation_service", self.namespace, self.version]
        
        if self.user_id:
            parts.append(f"user_{self.user_id}")
        
        parts.append(self.query_hash)
        
        return ":".join(parts)
    
    @classmethod
    def for_l0_pattern(cls, query_hash: str) -> "CacheKey":
        """Clé cache L0 patterns"""
        return cls(namespace="L0_PATTERN", query_hash=query_hash)
    
    @classmethod
    def for_l1_embedding(cls, query_hash: str, user_id: str) -> "CacheKey":
        """Clé cache L1 embeddings"""
        return cls(namespace="L1_LIGHTWEIGHT", query_hash=query_hash, user_id=user_id)
    
    @classmethod
    def for_l2_llm(cls, query_hash: str, user_id: str) -> "CacheKey":
        """Clé cache L2 LLM"""
        return cls(namespace="L2_LLM", query_hash=query_hash, user_id=user_id)

class CacheEntry(BaseModel):
    """Entrée cache avec métadonnées TTL"""
    
    key: CacheKey = Field(description="Clé cache structurée")
    value: IntentResult = Field(description="Résultat intention cached")
    ttl_seconds: int = Field(description="TTL en secondes")
    created_at: float = Field(default_factory=time.time, description="Timestamp création")
    hit_count: int = Field(default=0, description="Nombre d'accès")
    
    def is_expired(self) -> bool:
        """Vérifie si entrée cache expirée"""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def increment_hit(self):
        """Incrémente compteur accès"""
        self.hit_count += 1
    
    def get_remaining_ttl(self) -> int:
        """TTL restant en secondes"""
        elapsed = time.time() - self.created_at
        remaining = max(0, self.ttl_seconds - int(elapsed))
        return remaining

# ==========================================
# MODÈLES MÉTRIQUES ET MONITORING
# ==========================================

class LevelMetrics(BaseModel):
    """Métriques performance par niveau"""
    
    level: IntentLevel = Field(description="Niveau détection")
    total_requests: int = Field(default=0, description="Nombre total requêtes")
    successful_requests: int = Field(default=0, description="Requêtes réussies")
    failed_requests: int = Field(default=0, description="Requêtes échouées")
    
    total_latency_ms: float = Field(default=0.0, description="Latence totale cumulée")
    min_latency_ms: Optional[float] = Field(default=None, description="Latence minimum")
    max_latency_ms: Optional[float] = Field(default=None, description="Latence maximum")
    
    cache_hits: int = Field(default=0, description="Hits cache")
    cache_misses: int = Field(default=0, description="Misses cache")
    
    def get_success_rate(self) -> float:
        """Taux de succès"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_average_latency(self) -> float:
        """Latence moyenne"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests
    
    def get_cache_hit_rate(self) -> float:
        """Taux hit cache"""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.cache_hits / total_cache_requests
    
    def update_latency(self, latency_ms: float):
        """Met à jour statistiques latence"""
        self.total_latency_ms += latency_ms
        
        if self.min_latency_ms is None or latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        
        if self.max_latency_ms is None or latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms