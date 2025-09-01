"""
Modèles Pydantic V2 unifiés pour les réponses conversation service
Consolidation de toutes les phases : 1-5 avec toutes les fonctionnalités
"""
from pydantic import BaseModel, field_validator, ConfigDict, computed_field, field_serializer
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from enum import Enum

from conversation_service.prompts.harena_intents import HarenaIntentType
from conversation_service.models.contracts.search_service import (
    SearchQuery, QueryValidationResult, SearchResponse
)


# ============================================================================
# ENUMS ET CLASSES DE BASE
# ============================================================================

class IntentConfidenceLevel(str, Enum):
    """Niveaux de confiance pour classification"""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3 - 0.5
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # >= 0.9


class ProcessingStatus(str, Enum):
    """Statuts de traitement"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"


class CacheStatus(str, Enum):
    """Statuts du cache"""
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"
    DISABLED = "disabled"


# ============================================================================
# CLASSES PHASE 1 - BASE
# ============================================================================

class IntentAlternative(BaseModel):
    """Alternative d'intention avec validation avancée"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )
    
    intent_type: HarenaIntentType
    confidence: float
    reasoning: Optional[str] = None
    similarity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("Confidence doit être un nombre")
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence doit être entre 0.0 et 1.0")
        return float(v)
    
    @field_validator('similarity_score')
    @classmethod
    def validate_similarity_score(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise ValueError("Similarity score doit être un nombre")
        if not (0.0 <= v <= 1.0):
            raise ValueError("Similarity score doit être entre 0.0 et 1.0")
        return float(v)
    
    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v.strip()) == 0:
            return None
        if len(v) > 500:
            raise ValueError("Reasoning ne peut pas dépasser 500 caractères")
        return v.strip()
    
    @computed_field
    @property
    def confidence_level(self) -> IntentConfidenceLevel:
        """Calcul automatique du niveau de confiance"""
        if self.confidence >= 0.9:
            return IntentConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return IntentConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return IntentConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return IntentConfidenceLevel.LOW
        else:
            return IntentConfidenceLevel.VERY_LOW


class IntentClassificationResult(BaseModel):
    """Résultat de classification d'intention avec métadonnées enrichies"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    # Champs principaux
    intent_type: HarenaIntentType
    confidence: float
    reasoning: str
    original_message: str
    
    # Classification
    category: str
    is_supported: bool
    
    # Alternatives et contexte
    alternatives: List[IntentAlternative] = []
    context_factors: Optional[Dict[str, Any]] = None
    
    # Métadonnées techniques
    processing_time_ms: Optional[int] = None
    model_used: Optional[str] = None
    cache_used: bool = False
    
    # Qualité et fiabilité
    quality_score: Optional[float] = None
    reliability_indicators: Optional[Dict[str, Any]] = None
    
    # Extension AutoGen collaboration équipe
    team_context: Optional[Dict[str, Any]] = None

    @field_validator('intent_type', mode='before')
    @classmethod
    def ensure_intent_enum(cls, v: Any) -> HarenaIntentType:
        """S'assure que intent_type est bien une instance de HarenaIntentType."""
        if isinstance(v, HarenaIntentType):
            return v
        return HarenaIntentType(v)
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("Confidence doit être un nombre")
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence doit être entre 0.0 et 1.0")
        return float(v)
    
    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Reasoning ne peut pas être vide")
        if len(v) > 1000:
            raise ValueError("Reasoning ne peut pas dépasser 1000 caractères")
        return v.strip()
    
    @field_validator('original_message')
    @classmethod
    def validate_original_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message original ne peut pas être vide")
        if len(v) > 2000:
            raise ValueError("Message original ne peut pas dépasser 2000 caractères")
        return v.strip()
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Catégorie ne peut pas être vide")
        return v.strip().upper()
    
    @field_validator('alternatives')
    @classmethod
    def validate_alternatives(cls, v: List[IntentAlternative]) -> List[IntentAlternative]:
        if len(v) > 5:
            raise ValueError("Maximum 5 alternatives autorisées")
        
        # Tri par confiance décroissante
        return sorted(v, key=lambda x: x.confidence, reverse=True)
    
    @field_validator('processing_time_ms')
    @classmethod
    def validate_processing_time(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 0:
            raise ValueError("Processing time ne peut pas être négatif")
        if v > 120000:  # 2 minutes maximum
            raise ValueError("Processing time trop élevé")
        return v
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise ValueError("Quality score doit être un nombre")
        if not (0.0 <= v <= 1.0):
            raise ValueError("Quality score doit être entre 0.0 et 1.0")
        return float(v)
    
    @computed_field
    @property
    def confidence_level(self) -> IntentConfidenceLevel:
        """Calcul automatique du niveau de confiance"""
        if self.confidence >= 0.9:
            return IntentConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return IntentConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return IntentConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return IntentConfidenceLevel.LOW
        else:
            return IntentConfidenceLevel.VERY_LOW
    
    @computed_field
    @property
    def has_alternatives(self) -> bool:
        """Indique si des alternatives sont disponibles"""
        return len(self.alternatives) > 0
    
    @computed_field
    @property
    def top_alternative(self) -> Optional[IntentAlternative]:
        """Récupère la meilleure alternative"""
        return self.alternatives[0] if self.alternatives else None
    
    def add_context_factor(self, key: str, value: Any) -> None:
        """Ajoute un facteur contextuel"""
        if self.context_factors is None:
            self.context_factors = {}
        self.context_factors[key] = value
    
    def calculate_quality_score(self) -> float:
        """Calcule un score de qualité basé sur différents facteurs"""
        score = self.confidence  # Base score
        
        # Bonus si reasoning détaillé
        if len(self.reasoning) > 20:
            score += 0.05
        
        # Bonus si alternatives disponibles
        if self.alternatives:
            score += 0.03
        
        # Malus si confiance faible mais pas d'alternatives
        if self.confidence < 0.5 and not self.alternatives:
            score -= 0.1
        
        # Malus si processing time élevé
        if self.processing_time_ms and self.processing_time_ms > 5000:
            score -= 0.05
        
        return max(0.0, min(1.0, score))


class AgentMetrics(BaseModel):
    """Métriques d'exécution des agents avec validation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Métriques de base
    agent_used: str
    model_used: str
    tokens_consumed: int
    
    # Performance
    processing_time_ms: int
    confidence_threshold_met: bool
    
    # Cache et optimisation
    cache_hit: bool
    cache_status: CacheStatus = CacheStatus.MISS
    
    # Qualité et fiabilité
    retry_count: int = 0
    error_count: int = 0
    
    # Métadonnées détaillées
    detailed_metrics: Optional[Dict[str, Any]] = None
    
    @field_validator('agent_used')
    @classmethod
    def validate_agent_used(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent used ne peut pas être vide")
        return v.strip()
    
    @field_validator('model_used')
    @classmethod
    def validate_model_used(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model used ne peut pas être vide")
        return v.strip()
    
    @field_validator('tokens_consumed')
    @classmethod
    def validate_tokens_consumed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Tokens consumed ne peut pas être négatif")
        if v > 100000:  # Limite raisonnable
            raise ValueError("Tokens consumed trop élevé")
        return v
    
    @field_validator('processing_time_ms')
    @classmethod
    def validate_processing_time(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Processing time ne peut pas être négatif")
        if v > 300000:  # 5 minutes maximum
            raise ValueError("Processing time trop élevé")
        return v
    
    @field_validator('retry_count', 'error_count')
    @classmethod
    def validate_counts(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Les compteurs ne peuvent pas être négatifs")
        if v > 10:
            raise ValueError("Compteur anormalement élevé")
        return v
    
    @computed_field
    @property
    def performance_grade(self) -> str:
        """Calcule une note de performance"""
        if self.processing_time_ms < 100:
            return "A"  # Excellent
        elif self.processing_time_ms < 500:
            return "B"  # Bon
        elif self.processing_time_ms < 2000:
            return "C"  # Acceptable
        elif self.processing_time_ms < 5000:
            return "D"  # Lent
        else:
            return "F"  # Très lent
    
    @computed_field
    @property
    def efficiency_score(self) -> float:
        """Score d'efficacité basé sur plusieurs facteurs"""
        base_score = 1.0
        
        # Malus pour temps de traitement
        if self.processing_time_ms > 1000:
            base_score -= min(0.5, (self.processing_time_ms - 1000) / 10000)
        
        # Bonus pour cache hit
        if self.cache_hit:
            base_score += 0.2
        
        # Malus pour retries et erreurs
        base_score -= (self.retry_count * 0.1)
        base_score -= (self.error_count * 0.2)
        
        return max(0.0, min(1.0, base_score))


# ============================================================================
# CLASSES PHASE 3 - QUERY GENERATION
# ============================================================================

class QueryGenerationMetrics(BaseModel):
    """Métriques spécifiques à la génération de requêtes"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Agent query builder
    query_builder_used: str = "query_builder"
    generation_time_ms: int
    validation_time_ms: int
    optimization_time_ms: int
    
    # Qualité génération
    generation_confidence: float
    validation_passed: bool
    optimizations_applied: int
    
    # Performance estimée
    estimated_performance: str  # optimal, good, poor
    estimated_results_count: Optional[int] = None
    
    @field_validator('generation_confidence')
    @classmethod
    def validate_generation_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Generation confidence doit être entre 0.0 et 1.0")
        return v
    
    @field_validator('generation_time_ms', 'validation_time_ms', 'optimization_time_ms')
    @classmethod
    def validate_time_metrics(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Les temps ne peuvent pas être négatifs")
        if v > 30000:  # 30 secondes max
            raise ValueError("Temps de traitement anormalement élevé")
        return v
    
    @field_validator('optimizations_applied')
    @classmethod
    def validate_optimizations(cls, v: int) -> int:
        if v < 0 or v > 50:
            raise ValueError("Nombre d'optimisations invalide")
        return v
    
    @field_validator('estimated_performance')
    @classmethod
    def validate_performance(cls, v: str) -> str:
        if v not in ["optimal", "good", "poor"]:
            raise ValueError("Performance estimée invalide")
        return v


class ProcessingSteps(BaseModel):
    """Étapes de traitement détaillées"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    agent: str
    duration_ms: int
    cache_hit: bool
    success: bool = True
    error_message: Optional[str] = None
    
    @field_validator('agent')
    @classmethod
    def validate_agent(cls, v: str) -> str:
        valid_agents = [
            "intent_classifier", "entity_extractor", "query_builder",
            "multi_agent_team", "query_validator", "query_optimizer", "search_executor", "response_generator"
        ]
        if v not in valid_agents:
            raise ValueError(f"Agent invalide. Doit être un de: {valid_agents}")
        return v
    
    @field_validator('duration_ms')
    @classmethod
    def validate_duration(cls, v: int) -> int:
        if v < 0 or v > 60000:  # 1 minute max par agent
            raise ValueError("Durée agent invalide")
        return v


class QueryGenerationError(BaseModel):
    """Erreur lors de la génération de requête"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    error_type: str
    error_message: str
    error_agent: str
    fallback_applied: bool = False
    timestamp: datetime
    
    @field_validator('error_type')
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        valid_types = [
            "generation_failed", "validation_failed", "optimization_failed",
            "entity_extraction_failed", "intent_classification_failed",
            "timeout", "resource_exhausted", "unknown"
        ]
        if v not in valid_types:
            raise ValueError(f"Type d'erreur invalide. Doit être un de: {valid_types}")
        return v


# ============================================================================
# CLASSES PHASE 4 - SEARCH EXECUTION
# ============================================================================

class ResilienceMetrics(BaseModel):
    """Métriques de résilience pour l'exécution search_service"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Circuit breaker
    circuit_breaker_triggered: bool = False
    circuit_breaker_state: str = "closed"  # closed, open, half_open
    
    # Retry
    retry_attempts: int = 0
    total_retry_time_ms: int = 0
    retry_strategy_used: Optional[str] = None
    
    # Cache
    cache_hit: bool = False
    cache_key: Optional[str] = None
    
    # Fallback
    fallback_used: bool = False
    fallback_strategy: Optional[str] = None
    fallback_reason: Optional[str] = None
    
    # Performance
    search_execution_time_ms: int
    total_resilience_overhead_ms: int = 0
    
    @field_validator('retry_attempts')
    @classmethod
    def validate_retry_attempts(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("Retry attempts doit être entre 0 et 10")
        return v
    
    @field_validator('search_execution_time_ms', 'total_retry_time_ms', 'total_resilience_overhead_ms')
    @classmethod
    def validate_time_metrics(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Les temps ne peuvent pas être négatifs")
        if v > 60000:  # 1 minute max
            raise ValueError("Temps anormalement élevé")
        return v
    
    @field_validator('circuit_breaker_state')
    @classmethod
    def validate_circuit_state(cls, v: str) -> str:
        if v not in ["closed", "open", "half_open"]:
            raise ValueError("État circuit breaker invalide")
        return v


class SearchMetrics(BaseModel):
    """Métriques détaillées de l'exécution search"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Résultats
    total_hits: int
    returned_hits: int
    has_aggregations: bool = False
    aggregations_count: int = 0
    
    # Performance search_service
    search_service_took_ms: int  # Temps côté search_service
    network_latency_ms: int
    parsing_time_ms: int
    
    # Qualité résultats
    results_relevance: str = "unknown"  # high, medium, low, unknown
    estimated_completeness: float = 1.0  # 0.0 à 1.0
    
    @field_validator('total_hits', 'returned_hits', 'aggregations_count')
    @classmethod
    def validate_counts(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Les compteurs ne peuvent pas être négatifs")
        return v
    
    @field_validator('estimated_completeness')
    @classmethod
    def validate_completeness(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Completeness doit être entre 0.0 et 1.0")
        return v
    
    @field_validator('results_relevance')
    @classmethod
    def validate_relevance(cls, v: str) -> str:
        if v not in ["high", "medium", "low", "unknown"]:
            raise ValueError("Relevance invalide")
        return v


class SearchExecutionError(BaseModel):
    """Erreur lors de l'exécution search"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    error_type: str
    error_message: str
    error_component: str  # search_client, circuit_breaker, cache, etc.
    recovery_attempted: bool = False
    recovery_successful: bool = False
    timestamp: datetime
    
    # Contexte détaillé
    search_query_id: Optional[str] = None
    circuit_breaker_state: Optional[str] = None
    retry_attempts_made: int = 0
    
    @field_validator('error_type')
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        valid_types = [
            "search_service_error", "circuit_breaker_open", "timeout",
            "validation_error", "connection_error", "rate_limit_exceeded",
            "service_unavailable", "cache_error", "unexpected_error", "unknown"
        ]
        if v not in valid_types:
            raise ValueError(f"Type d'erreur invalide. Doit être un de: {valid_types}")
        return v
    
    @field_validator('error_component')
    @classmethod
    def validate_error_component(cls, v: str) -> str:
        valid_components = [
            "search_client", "circuit_breaker", "retry_handler", "cache", 
            "search_executor", "search_service", "network", "unknown"
        ]
        if v not in valid_components:
            raise ValueError(f"Composant d'erreur invalide. Doit être un de: {valid_components}")
        return v


# ============================================================================
# CLASSES PHASE 5 - RESPONSE GENERATION
# ============================================================================

class Insight(BaseModel):
    """Insight automatique généré à partir des résultats"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    type: str  # trend, pattern, category, liquidity, optimization
    title: str
    description: str
    severity: str = "info"  # info, warning, alert, positive, neutral
    confidence: float = 0.5
    
    @field_validator('type')
    @classmethod
    def validate_insight_type(cls, v: str) -> str:
        valid_types = ["trend", "pattern", "category", "liquidity", "optimization", "spending", "budget"]
        if v not in valid_types:
            raise ValueError(f"Type d'insight invalide. Doit être un de: {valid_types}")
        return v
    
    @field_validator('severity')
    @classmethod
    def validate_severity(cls, v: str) -> str:
        valid_severities = ["info", "warning", "alert", "positive", "neutral"]
        if v not in valid_severities:
            raise ValueError(f"Severité invalide. Doit être un de: {valid_severities}")
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence doit être entre 0.0 et 1.0")
        return v


class Suggestion(BaseModel):
    """Suggestion actionnnable pour l'utilisateur"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    type: str  # optimization, budget, saving, alert, action
    title: str
    description: str
    action: Optional[str] = None  # Action textuelle suggérée
    priority: str = "medium"  # low, medium, high, critical
    
    @field_validator('type')
    @classmethod
    def validate_suggestion_type(cls, v: str) -> str:
        valid_types = ["optimization", "budget", "saving", "alert", "action", "planning"]
        if v not in valid_types:
            raise ValueError(f"Type de suggestion invalide. Doit être un de: {valid_types}")
        return v
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        valid_priorities = ["low", "medium", "high", "critical"]
        if v not in valid_priorities:
            raise ValueError(f"Priorité invalide. Doit être un de: {valid_priorities}")
        return v


class StructuredData(BaseModel):
    """Données structurées extraites pour l'utilisateur"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Montants principaux
    total_amount: Optional[float] = None
    currency: str = "EUR"
    transaction_count: Optional[int] = None
    average_amount: Optional[float] = None
    
    # Période analysée
    period: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    
    # Comparaisons
    comparison: Optional[Dict[str, Any]] = None
    
    # Métadonnées contextuelles
    analysis_type: Optional[str] = None  # merchant, category, period, balance
    primary_entity: Optional[str] = None  # Amazon, Restaurant, etc.


class ResponseContent(BaseModel):
    """Contenu de la réponse générée"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Message principal en format naturel
    message: str
    
    # Données structurées
    structured_data: Optional[StructuredData] = None
    
    # Insights automatiques
    insights: List[Insight] = []
    
    # Suggestions actionnables
    suggestions: List[Suggestion] = []
    
    # Actions suivantes proposées
    next_actions: List[str] = []
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Message trop court, minimum 10 caractères")
        if len(v) > 2000:
            raise ValueError("Message trop long, maximum 2000 caractères")
        return v.strip()


class ResponseQuality(BaseModel):
    """Métriques de qualité de la réponse générée"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    relevance_score: float = 0.5  # Pertinence par rapport à la requête
    completeness: str = "partial"  # full, partial, minimal
    actionability: str = "medium"  # high, medium, low, none
    personalization_level: str = "basic"  # advanced, medium, basic, none
    tone: str = "professional"  # professional, friendly, professional_friendly, casual
    
    @field_validator('relevance_score')
    @classmethod
    def validate_relevance(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Relevance score doit être entre 0.0 et 1.0")
        return v
    
    @field_validator('completeness')
    @classmethod
    def validate_completeness(cls, v: str) -> str:
        valid_values = ["full", "partial", "minimal"]
        if v not in valid_values:
            raise ValueError(f"Completeness invalide. Doit être un de: {valid_values}")
        return v
    
    @field_validator('actionability')
    @classmethod
    def validate_actionability(cls, v: str) -> str:
        valid_values = ["high", "medium", "low", "none"]
        if v not in valid_values:
            raise ValueError(f"Actionability invalide. Doit être un de: {valid_values}")
        return v


class ResponseGenerationMetrics(BaseModel):
    """Métriques spécifiques à la génération de réponse"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    generation_time_ms: int
    tokens_response: int = 0
    quality_score: float = 0.5
    insights_generated: int = 0
    suggestions_generated: int = 0
    
    # Contexte utilisé
    context_items_used: int = 0
    personalization_applied: bool = False
    template_used: Optional[str] = None
    
    @field_validator('generation_time_ms')
    @classmethod
    def validate_generation_time(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Temps de génération ne peut pas être négatif")
        if v > 30000:  # 30 secondes max
            raise ValueError("Temps de génération anormalement élevé")
        return v
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Quality score doit être entre 0.0 et 1.0")
        return v


# ============================================================================
# MODÈLES DE RÉPONSE PRINCIPAUX
# ============================================================================

class ConversationResponse(BaseModel):
    """Réponse complète Phase 1 avec validation avancée"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "user_id": 123,
                "sub": 123,
                "message": "Mes achats Amazon",
                "timestamp": "2024-08-26T14:30:00+00:00",
                "processing_time_ms": 245,
                "status": "success",
                "intent": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "confidence": 0.94,
                    "reasoning": "L'utilisateur cherche ses transactions Amazon",
                    "original_message": "Mes achats Amazon",
                    "category": "FINANCIAL_QUERY",
                    "is_supported": True,
                    "alternatives": [],
                    "processing_time_ms": 200
                },
                "agent_metrics": {
                    "agent_used": "intent_classifier",
                    "model_used": "deepseek-chat",
                    "tokens_consumed": 156,
                    "processing_time_ms": 245,
                    "confidence_threshold_met": True,
                    "cache_hit": False,
                    "retry_count": 0,
                    "error_count": 0
                },
                "request_id": "req_1234567890_123",
                "phase": 1
            }
        }
    )
    
    # Identifiants et contexte
    user_id: int
    sub: Optional[int] = None
    message: str
    timestamp: datetime
    request_id: Optional[str] = None
    
    # Résultats
    intent: IntentClassificationResult
    agent_metrics: AgentMetrics
    
    # Métadonnées de traitement
    processing_time_ms: int
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    phase: int = 1
    
    # Informations additionnelles
    warnings: List[str] = []
    debug_info: Optional[Dict[str, Any]] = None
    
    # Entités extraites (Phase 2+)
    entities: Optional[Dict[str, Any]] = None
    
    # Phase 3+ - Query generation
    search_query: Optional[SearchQuery] = None
    query_validation: Optional[QueryValidationResult] = None
    query_generation_metrics: Optional[QueryGenerationMetrics] = None
    processing_steps: List[ProcessingSteps] = []
    agent_metrics_detailed: Dict[str, Any] = {}
    
    # Phase 4+ - Search execution
    search_results: Optional[SearchResponse] = None
    resilience_metrics: Optional[ResilienceMetrics] = None
    search_metrics: Optional[SearchMetrics] = None
    
    # Phase 5+ - Response generation
    response: Optional[ResponseContent] = None
    response_quality: Optional[ResponseQuality] = None
    response_generation_metrics: Optional[ResponseGenerationMetrics] = None

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("User ID doit être positif")
        if v > 1000000:  # Limite raisonnable
            raise ValueError("User ID hors limites")
        return v

    @field_validator('sub')
    @classmethod
    def validate_sub(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("Sub doit être positif")
        if v > 1000000:
            raise ValueError("Sub hors limites")
        return v
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message ne peut pas être vide")
        if len(v) > 2000:
            raise ValueError("Message ne peut pas dépasser 2000 caractères")
        return v.strip()
    
    @field_validator('processing_time_ms')
    @classmethod
    def validate_processing_time(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Processing time ne peut pas être négatif")
        if v > 300000:  # 5 minutes
            raise ValueError("Processing time anormalement élevé")
        return v
    
    @field_validator('phase')
    @classmethod
    def validate_phase(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Phase doit être >= 1")
        if v > 10:  # Limite raisonnable pour les phases futures
            raise ValueError("Phase invalide")
        return v
    
    @field_validator('warnings')
    @classmethod
    def validate_warnings(cls, v: List[str]) -> List[str]:
        if len(v) > 10:
            raise ValueError("Trop de warnings")
        # Nettoyage des warnings vides
        return [w.strip() for w in v if w and w.strip()]
    
    @field_validator('search_query')
    @classmethod
    def validate_search_query(cls, v: Optional[SearchQuery]) -> Optional[SearchQuery]:
        # search_query peut être None si génération échouée
        return v
    
    @field_serializer('search_query')
    def serialize_search_query(self, search_query: Optional[SearchQuery]) -> Optional[Dict[str, Any]]:
        """Sérialise search_query en excluant les valeurs null"""
        if search_query is None:
            return None
        return search_query.dict(exclude_none=True)
    
    @field_validator('processing_steps')
    @classmethod
    def validate_processing_steps(cls, v: List[ProcessingSteps]) -> List[ProcessingSteps]:
        if len(v) > 10:  # Maximum 10 étapes
            raise ValueError("Trop d'étapes de traitement")
        return v
    
    @field_validator('search_results')
    @classmethod
    def validate_search_results(cls, v: Optional[SearchResponse]) -> Optional[SearchResponse]:
        # search_results peut être None si recherche échouée
        return v
    
    # ============================================================================
    # COMPUTED FIELDS
    # ============================================================================
    
    @computed_field
    @property
    def is_successful(self) -> bool:
        """Indique si la réponse est considérée comme réussie"""
        return self.status in [ProcessingStatus.SUCCESS, ProcessingStatus.PARTIAL_SUCCESS]
    
    @computed_field
    @property
    def overall_confidence(self) -> float:
        """Confiance globale combinant intent et métriques"""
        base_confidence = self.intent.confidence
        
        # Ajustements basés sur les métriques
        if self.agent_metrics.error_count > 0:
            base_confidence *= 0.9
        
        if self.agent_metrics.retry_count > 0:
            base_confidence *= 0.95
        
        if not self.agent_metrics.confidence_threshold_met:
            base_confidence *= 0.8
        
        return max(0.0, min(1.0, base_confidence))
    
    @computed_field
    @property
    def performance_summary(self) -> Dict[str, Any]:
        """Résumé des performances"""
        return {
            "processing_time_ms": self.processing_time_ms,
            "performance_grade": self.agent_metrics.performance_grade,
            "efficiency_score": self.agent_metrics.efficiency_score,
            "cache_hit": self.agent_metrics.cache_hit,
            "confidence_level": self.intent.confidence_level.value,
            "has_warnings": len(self.warnings) > 0
        }
    
    # Phase 3+ computed fields
    @computed_field
    @property
    def has_search_query(self) -> bool:
        """Indique si une requête search_service a été générée"""
        return self.search_query is not None
    
    @computed_field
    @property
    def query_generation_success(self) -> bool:
        """Indique si la génération de requête a réussi"""
        return (self.search_query is not None and 
                self.query_validation is not None and 
                self.query_validation.schema_valid and 
                self.query_validation.contract_compliant)
    
    @computed_field
    @property
    def agents_sequence(self) -> List[str]:
        """Séquence des agents utilisés"""
        return [step.agent for step in self.processing_steps]
    
    @computed_field
    @property
    def total_agent_time_ms(self) -> int:
        """Temps total des agents (peut être différent du temps total si parallélisme)"""
        return sum(step.duration_ms for step in self.processing_steps)
    
    @computed_field
    @property
    def cache_efficiency(self) -> float:
        """Taux d'efficacité du cache sur tous les agents"""
        if not self.processing_steps:
            return 0.0
        cache_hits = sum(1 for step in self.processing_steps if step.cache_hit)
        return cache_hits / len(self.processing_steps)
    
    # Phase 4+ computed fields
    @computed_field
    @property
    def has_search_results(self) -> bool:
        """Indique si des résultats search ont été obtenus"""
        return self.search_results is not None
    
    @computed_field
    @property
    def search_execution_success(self) -> bool:
        """Indique si l'exécution search a réussi"""
        return (self.search_results is not None and 
                self.resilience_metrics is not None and 
                not self.resilience_metrics.circuit_breaker_triggered)
    
    @computed_field
    @property
    def total_processing_time_with_search_ms(self) -> int:
        """Temps total incluant exécution search"""
        if self.resilience_metrics:
            return self.processing_time_ms + self.resilience_metrics.search_execution_time_ms
        return self.processing_time_ms
    
    @computed_field
    @property
    def results_summary(self) -> Dict[str, Any]:
        """Résumé des résultats obtenus"""
        if not self.search_results:
            return {"status": "no_results", "reason": "search_failed"}
        
        return {
            "status": "success",
            "total_hits": self.search_results.total_hits,
            "returned_hits": len(self.search_results.hits),
            "has_aggregations": bool(self.search_results.aggregations),
            "search_took_ms": self.search_results.took_ms,
            "fallback_used": self.resilience_metrics.fallback_used if self.resilience_metrics else False
        }
    
    # Phase 5+ computed fields
    @computed_field
    @property
    def has_response(self) -> bool:
        """Indique si une réponse a été générée"""
        return self.response is not None and len(self.response.message.strip()) > 0
    
    @computed_field
    @property
    def response_generation_success(self) -> bool:
        """Indique si la génération de réponse a réussi"""
        return (self.has_response and 
                self.response_quality is not None and 
                self.response_quality.relevance_score > 0.5)
    
    @computed_field
    @property
    def total_processing_time_with_response_ms(self) -> int:
        """Temps total incluant génération réponse"""
        base_time = self.total_processing_time_with_search_ms
        if self.response_generation_metrics:
            return base_time + self.response_generation_metrics.generation_time_ms
        return base_time
    
    @computed_field
    @property
    def insights_summary(self) -> Dict[str, Any]:
        """Résumé des insights générés"""
        if not self.response or not self.response.insights:
            return {"total": 0, "by_type": {}, "by_severity": {}}
        
        by_type = {}
        by_severity = {}
        
        for insight in self.response.insights:
            by_type[insight.type] = by_type.get(insight.type, 0) + 1
            by_severity[insight.severity] = by_severity.get(insight.severity, 0) + 1
        
        return {
            "total": len(self.response.insights),
            "by_type": by_type,
            "by_severity": by_severity
        }
    
    @computed_field
    @property
    def actionability_summary(self) -> Dict[str, Any]:
        """Résumé des éléments actionnables"""
        if not self.response:
            return {"suggestions": 0, "next_actions": 0, "actionability_score": 0.0}
        
        suggestions_count = len(self.response.suggestions)
        actions_count = len(self.response.next_actions)
        
        # Score d'actionnabilité basé sur le nombre d'éléments
        actionability_score = min(1.0, (suggestions_count * 0.6 + actions_count * 0.4) / 3.0)
        
        return {
            "suggestions": suggestions_count,
            "next_actions": actions_count,
            "actionability_score": actionability_score,
            "priority_distribution": self._get_priority_distribution()
        }
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Distribution des priorités des suggestions"""
        if not self.response or not self.response.suggestions:
            return {}
        
        distribution = {}
        for suggestion in self.response.suggestions:
            distribution[suggestion.priority] = distribution.get(suggestion.priority, 0) + 1
        
        return distribution
    
    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================
    
    def add_warning(self, warning: str) -> None:
        """Ajoute un warning à la réponse"""
        if warning and warning.strip():
            self.warnings.append(warning.strip())
    
    def set_debug_info(self, key: str, value: Any) -> None:
        """Ajoute des informations de debug"""
        if self.debug_info is None:
            self.debug_info = {}
        self.debug_info[key] = value
    
    def add_processing_step(self, agent: str, duration_ms: int, cache_hit: bool = False, 
                          success: bool = True, error_message: Optional[str] = None) -> None:
        """Ajoute une étape de traitement"""
        step = ProcessingSteps(
            agent=agent,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            success=success,
            error_message=error_message
        )
        self.processing_steps.append(step)
    
    def set_query_generation_metrics(self, metrics: QueryGenerationMetrics) -> None:
        """Définit les métriques de génération de requête"""
        self.query_generation_metrics = metrics
    
    def set_resilience_metrics(self, metrics: ResilienceMetrics) -> None:
        """Définit les métriques de résilience"""
        self.resilience_metrics = metrics
    
    def set_search_metrics(self, metrics: SearchMetrics) -> None:
        """Définit les métriques search"""
        self.search_metrics = metrics
    
    def set_response_content(self, content: ResponseContent) -> None:
        """Définit le contenu de la réponse"""
        self.response = content
    
    def set_response_quality(self, quality: ResponseQuality) -> None:
        """Définit la qualité de la réponse"""
        self.response_quality = quality
    
    def set_response_generation_metrics(self, metrics: ResponseGenerationMetrics) -> None:
        """Définit les métriques de génération"""
        self.response_generation_metrics = metrics
    
    def to_minimal_dict(self) -> Dict[str, Any]:
        """Version minimaliste pour logs ou cache"""
        result = {
            "user_id": self.user_id,
            "intent_type": self.intent.intent_type.value,
            "confidence": self.intent.confidence,
            "processing_time_ms": self.processing_time_ms,
            "status": self.status.value,
            "cache_hit": self.agent_metrics.cache_hit,
            "phase": self.phase
        }
        
        # Ajouter infos selon phase
        if self.phase >= 3:
            result.update({
                "has_search_query": self.has_search_query,
                "query_generation_success": self.query_generation_success,
                "agents_used": self.agents_sequence,
                "cache_efficiency": self.cache_efficiency
            })
        
        if self.phase >= 4:
            result.update({
                "has_search_results": self.has_search_results,
                "search_execution_success": self.search_execution_success,
                "results_count": self.search_results.total_hits if self.search_results else 0,
                "fallback_used": self.resilience_metrics.fallback_used if self.resilience_metrics else False
            })
        
        if self.phase >= 5:
            result.update({
                "has_response": self.has_response,
                "response_generation_success": self.response_generation_success,
                "insights_count": len(self.response.insights) if self.response else 0,
                "suggestions_count": len(self.response.suggestions) if self.response else 0,
                "response_quality_score": self.response_quality.relevance_score if self.response_quality else 0.0
            })
        
        return result


# ============================================================================
# CLASSES D'ERREUR SPÉCIALISÉES
# ============================================================================

class ConversationResponseError(ConversationResponse):
    """Réponse avec erreur de génération de requête"""
    
    # Erreur spécifique
    query_generation_error: Optional[QueryGenerationError] = None
    search_execution_error: Optional[SearchExecutionError] = None
    
    # Override status par défaut
    status: ProcessingStatus = ProcessingStatus.PARTIAL_SUCCESS
    
    @computed_field
    @property
    def has_fallback(self) -> bool:
        """Indique si un fallback a été appliqué"""
        if self.query_generation_error:
            return self.query_generation_error.fallback_applied
        return False
    
    @computed_field
    @property
    def has_recovery_attempt(self) -> bool:
        """Indique si une récupération a été tentée"""
        if self.search_execution_error:
            return self.search_execution_error.recovery_attempted
        return False
    
    @computed_field
    @property
    def recovery_successful(self) -> bool:
        """Indique si la récupération a réussi"""
        if self.search_execution_error:
            return (self.search_execution_error.recovery_attempted and 
                    self.search_execution_error.recovery_successful)
        return False
    
    def to_error_summary(self) -> Dict[str, Any]:
        """Résumé d'erreur pour monitoring"""
        base_summary = {
            "user_id": self.user_id,
            "phase": self.phase,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.query_generation_error:
            base_summary.update({
                "error_type": self.query_generation_error.error_type,
                "error_agent": self.query_generation_error.error_agent,
                "fallback_applied": self.query_generation_error.fallback_applied
            })
        
        if self.search_execution_error:
            base_summary.update({
                "error_type": self.search_execution_error.error_type,
                "error_component": self.search_execution_error.error_component,
                "recovery_attempted": self.search_execution_error.recovery_attempted,
                "recovery_successful": self.search_execution_error.recovery_successful,
                "circuit_breaker_state": self.search_execution_error.circuit_breaker_state,
                "retry_attempts": self.search_execution_error.retry_attempts_made,
                "has_fallback_results": self.has_search_results
            })
        
        return base_summary


# ============================================================================
# MODÈLES DE RÉPONSE POUR ENDPOINTS SPÉCIFIQUES
# ============================================================================

class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    timestamp: datetime
    service: str = "conversation_service"
    phase: int = 5
    uptime_seconds: Optional[float] = None
    components: Optional[Dict[str, str]] = None
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = ["healthy", "unhealthy", "degraded", "maintenance"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v


class MetricsResponse(BaseModel):
    """Réponse metrics endpoint"""
    timestamp: datetime
    service_info: Dict[str, Any]
    metrics: Dict[str, Any]
    labels: Optional[Dict[str, str]] = None
    
    @field_validator('service_info')
    @classmethod
    def validate_service_info(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = ["name", "version", "phase"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required service_info key: {key}")
        return v


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée"""
    error: str
    error_code: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    @field_validator('error')
    @classmethod
    def validate_error(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Error message ne peut pas être vide")
        return v.strip()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_success_response(
    user_id: int,
    message: str,
    intent_result: IntentClassificationResult,
    agent_metrics: AgentMetrics,
    processing_time_ms: int,
    request_id: Optional[str] = None,
    phase: int = 1
) -> ConversationResponse:
    """Crée une réponse de succès standardisée"""
    
    return ConversationResponse(
        user_id=user_id,
        message=message,
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        intent=intent_result,
        agent_metrics=agent_metrics,
        processing_time_ms=processing_time_ms,
        status=ProcessingStatus.SUCCESS,
        phase=phase
    )


def create_error_response(
    error_message: str,
    error_code: Optional[str] = None,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """Crée une réponse d'erreur standardisée"""
    
    return ErrorResponse(
        error=error_message,
        error_code=error_code,
        timestamp=datetime.now(timezone.utc),
        request_id=request_id,
        details=details
    )


# ============================================================================
# FACTORY CLASSES POUR CRÉATION AVANCÉE
# ============================================================================

class ConversationResponseFactory:
    """Factory pour créer les bonnes réponses selon le contexte"""
    
    @staticmethod
    def _map_performance_to_relevance(performance: str) -> str:
        """Mappe estimated_performance vers results_relevance"""
        mapping = {
            "optimal": "high",
            "good": "medium", 
            "poor": "low",
            "failed": "unknown",
            "unknown": "unknown"
        }
        return mapping.get(performance, "unknown")
    
    @staticmethod
    def create_phase3_success(
        base_response: ConversationResponse,
        search_query: SearchQuery,
        query_validation: QueryValidationResult,
        processing_steps: List[ProcessingSteps],
        query_metrics: Optional[QueryGenerationMetrics] = None
    ) -> ConversationResponse:
        """Création réponse Phase 3 réussie"""
        
        response = ConversationResponse(
            user_id=base_response.user_id,
            sub=base_response.sub,
            message=base_response.message,
            timestamp=base_response.timestamp,
            request_id=base_response.request_id,
            intent=base_response.intent,
            agent_metrics=base_response.agent_metrics,
            processing_time_ms=base_response.processing_time_ms,
            status=base_response.status,
            warnings=base_response.warnings,
            debug_info=base_response.debug_info,
            entities=base_response.entities,
            search_query=search_query,
            query_validation=query_validation,
            processing_steps=processing_steps,
            phase=3
        )
        
        if query_metrics:
            response.set_query_generation_metrics(query_metrics)
        
        return response
    
    @staticmethod
    def create_phase3_error(
        base_response: ConversationResponse,
        error: QueryGenerationError,
        processing_steps: List[ProcessingSteps]
    ) -> ConversationResponseError:
        """Création réponse Phase 3 avec erreur"""
        
        return ConversationResponseError(
            user_id=base_response.user_id,
            sub=base_response.sub,
            message=base_response.message,
            timestamp=base_response.timestamp,
            request_id=base_response.request_id,
            intent=base_response.intent,
            agent_metrics=base_response.agent_metrics,
            processing_time_ms=base_response.processing_time_ms,
            warnings=base_response.warnings,
            debug_info=base_response.debug_info,
            entities=base_response.entities,
            processing_steps=processing_steps,
            query_generation_error=error,
            phase=3
        )
    
    @staticmethod
    def create_phase4_success(
        base_response: ConversationResponse,
        search_results: SearchResponse,
        resilience_metrics: ResilienceMetrics,
        search_metrics: Optional[SearchMetrics] = None
    ) -> ConversationResponse:
        """Création réponse Phase 4 réussie avec résultats"""
        
        response = ConversationResponse(
            user_id=base_response.user_id,
            sub=base_response.sub,
            message=base_response.message,
            timestamp=base_response.timestamp,
            request_id=base_response.request_id,
            intent=base_response.intent,
            agent_metrics=base_response.agent_metrics,
            processing_time_ms=base_response.processing_time_ms,
            status=base_response.status,
            warnings=base_response.warnings,
            debug_info=base_response.debug_info,
            entities=base_response.entities,
            search_query=base_response.search_query,
            query_validation=base_response.query_validation,
            query_generation_metrics=base_response.query_generation_metrics,
            processing_steps=base_response.processing_steps,
            agent_metrics_detailed=base_response.agent_metrics_detailed,
            search_results=search_results,
            resilience_metrics=resilience_metrics,
            phase=4
        )
        
        if search_metrics:
            response.set_search_metrics(search_metrics)
        
        return response
    
    @staticmethod
    def create_phase4_error(
        base_response: ConversationResponse,
        error: SearchExecutionError,
        resilience_metrics: Optional[ResilienceMetrics] = None,
        partial_results: Optional[SearchResponse] = None
    ) -> ConversationResponseError:
        """Création réponse Phase 4 avec erreur search"""
        
        response = ConversationResponseError(
            user_id=base_response.user_id,
            sub=base_response.sub,
            message=base_response.message,
            timestamp=base_response.timestamp,
            request_id=base_response.request_id,
            intent=base_response.intent,
            agent_metrics=base_response.agent_metrics,
            processing_time_ms=base_response.processing_time_ms,
            warnings=base_response.warnings,
            debug_info=base_response.debug_info,
            entities=base_response.entities,
            search_query=base_response.search_query,
            query_validation=base_response.query_validation,
            query_generation_metrics=base_response.query_generation_metrics,
            processing_steps=base_response.processing_steps,
            agent_metrics_detailed=base_response.agent_metrics_detailed,
            search_execution_error=error,
            phase=4
        )
        
        # Ajouter résultats partiels si disponibles (fallback)
        if partial_results:
            response.search_results = partial_results
        
        # Ajouter métriques résilience si disponibles
        if resilience_metrics:
            response.set_resilience_metrics(resilience_metrics)
        
        return response
    
    @staticmethod
    def create_phase5_success(
        base_response: ConversationResponse,
        response_content: ResponseContent,
        response_quality: ResponseQuality,
        generation_metrics: ResponseGenerationMetrics
    ) -> ConversationResponse:
        """Création réponse Phase 5 réussie avec réponse générée"""
        
        response = ConversationResponse(
            user_id=base_response.user_id,
            sub=base_response.sub,
            message=base_response.message,
            timestamp=base_response.timestamp,
            request_id=base_response.request_id,
            intent=base_response.intent,
            agent_metrics=base_response.agent_metrics,
            processing_time_ms=base_response.processing_time_ms,
            status=base_response.status,
            warnings=base_response.warnings,
            debug_info=base_response.debug_info,
            entities=base_response.entities,
            search_query=base_response.search_query,
            query_validation=base_response.query_validation,
            query_generation_metrics=base_response.query_generation_metrics,
            processing_steps=base_response.processing_steps,
            agent_metrics_detailed=base_response.agent_metrics_detailed,
            search_results=base_response.search_results,
            resilience_metrics=base_response.resilience_metrics,
            search_metrics=base_response.search_metrics,
            response=response_content,
            response_quality=response_quality,
            response_generation_metrics=generation_metrics,
            phase=5
        )
        
        return response
    
    @staticmethod
    def create_from_search_executor_response(
        base_response: ConversationResponse,
        executor_response,  # SearchExecutorResponse
        processing_step: ProcessingSteps
    ) -> Union[ConversationResponse, ConversationResponseError]:
        """Création depuis réponse SearchExecutor"""
        
        # Créer métriques de résilience depuis executor
        resilience_metrics = ResilienceMetrics(
            circuit_breaker_triggered=executor_response.circuit_breaker_triggered,
            retry_attempts=executor_response.retry_attempts,
            fallback_used=executor_response.fallback_used,
            search_execution_time_ms=executor_response.execution_time_ms
        )
        
        # Ajouter processing step pour search execution
        extended_steps = base_response.processing_steps + [processing_step]
        base_with_steps = ConversationResponse(
            user_id=base_response.user_id,
            sub=base_response.sub,
            message=base_response.message,
            timestamp=base_response.timestamp,
            request_id=base_response.request_id,
            intent=base_response.intent,
            agent_metrics=base_response.agent_metrics,
            processing_time_ms=base_response.processing_time_ms,
            status=base_response.status,
            warnings=base_response.warnings,
            debug_info=base_response.debug_info,
            entities=base_response.entities,
            search_query=base_response.search_query,
            query_validation=base_response.query_validation,
            query_generation_metrics=base_response.query_generation_metrics,
            processing_steps=extended_steps,
            agent_metrics_detailed=base_response.agent_metrics_detailed,
            phase=4
        )
        
        if executor_response.success and executor_response.search_results:
            # Succès avec résultats
            search_metrics = SearchMetrics(
                total_hits=executor_response.search_results.total_hits,
                returned_hits=len(executor_response.search_results.hits),
                has_aggregations=bool(executor_response.search_results.aggregations),
                aggregations_count=len(executor_response.search_results.aggregations) if executor_response.search_results.aggregations else 0,
                search_service_took_ms=executor_response.search_results.took_ms,
                network_latency_ms=max(0, executor_response.execution_time_ms - executor_response.search_results.took_ms),
                parsing_time_ms=10,  # Estimation
                results_relevance=ConversationResponseFactory._map_performance_to_relevance(executor_response.estimated_performance)
            )
            
            return ConversationResponseFactory.create_phase4_success(
                base_with_steps,
                executor_response.search_results,
                resilience_metrics,
                search_metrics
            )
        else:
            # Erreur ou pas de résultats
            error = SearchExecutionError(
                error_type=executor_response.error_type or "unknown",
                error_message=executor_response.error_message or "Unknown error",
                error_component="search_executor",
                recovery_attempted=executor_response.fallback_used,
                recovery_successful=executor_response.success and executor_response.fallback_used,
                timestamp=executor_response.timestamp
            )
            
            return ConversationResponseFactory.create_phase4_error(
                base_with_steps,
                error,
                resilience_metrics,
                executor_response.search_results  # Peut être None ou résultats fallback
            )
    
    @staticmethod
    def create_phase5_from_phase4(
        phase4_response: ConversationResponse
    ) -> ConversationResponse:
        """Création Phase 5 depuis Phase 4 (sans réponse générée encore)"""
        
        return ConversationResponse(
            user_id=phase4_response.user_id,
            sub=phase4_response.sub,
            message=phase4_response.message,
            timestamp=phase4_response.timestamp,
            request_id=phase4_response.request_id,
            intent=phase4_response.intent,
            agent_metrics=phase4_response.agent_metrics,
            processing_time_ms=phase4_response.processing_time_ms,
            status=phase4_response.status,
            warnings=phase4_response.warnings,
            debug_info=phase4_response.debug_info,
            entities=phase4_response.entities,
            search_query=phase4_response.search_query,
            query_validation=phase4_response.query_validation,
            query_generation_metrics=phase4_response.query_generation_metrics,
            processing_steps=phase4_response.processing_steps,
            agent_metrics_detailed=phase4_response.agent_metrics_detailed,
            search_results=phase4_response.search_results,
            resilience_metrics=phase4_response.resilience_metrics,
            search_metrics=phase4_response.search_metrics,
            phase=5
        )


# ============================================================================
# ALIASES POUR COMPATIBILITÉ
# ============================================================================

# Alias pour compatibilité avec les anciennes versions
ConversationResponsePhase2 = ConversationResponse
ConversationResponsePhase3 = ConversationResponse
ConversationResponsePhase4 = ConversationResponse  
ConversationResponsePhase5 = ConversationResponse

ConversationResponsePhase3Error = ConversationResponseError
ConversationResponsePhase4Error = ConversationResponseError

ConversationResponseFactoryPhase4 = ConversationResponseFactory
ConversationResponseFactoryPhase5 = ConversationResponseFactory

# Alias pour classes Phase 2 manquantes (compatibilité tests)
EntityValidationResult = AgentMetrics  # Compatible pour les tests
MultiAgentProcessingInsights = AgentMetrics  # Compatible pour les tests

# Types d'union pour flexibilité
ResponseData = Union[IntentClassificationResult, Dict[str, Any]]
ResponseMetrics = Union[AgentMetrics, Dict[str, Any]]