"""
Modèles Pydantic V2 optimisés pour les réponses conversation service
"""
from pydantic import BaseModel, field_validator, ConfigDict, computed_field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from enum import Enum
from conversation_service.prompts.harena_intents import HarenaIntentType


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
        if v > 60000:  # 60 secondes maximum
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
    
    def add_warning(self, warning: str) -> None:
        """Ajoute un warning à la réponse"""
        if warning and warning.strip():
            self.warnings.append(warning.strip())
    
    def set_debug_info(self, key: str, value: Any) -> None:
        """Ajoute des informations de debug"""
        if self.debug_info is None:
            self.debug_info = {}
        self.debug_info[key] = value
    
    def to_minimal_dict(self) -> Dict[str, Any]:
        """Version minimaliste pour logs ou cache"""
        return {
            "user_id": self.user_id,
            "intent_type": self.intent.intent_type.value,
            "confidence": self.intent.confidence,
            "processing_time_ms": self.processing_time_ms,
            "status": self.status.value,
            "cache_hit": self.agent_metrics.cache_hit
        }


# Types d'union pour flexibilité
ResponseData = Union[IntentClassificationResult, Dict[str, Any]]
ResponseMetrics = Union[AgentMetrics, Dict[str, Any]]

# Modèles de réponse pour endpoints spécifiques
class HealthResponse(BaseModel):
    """Réponse health check"""
    status: str
    timestamp: datetime
    service: str = "conversation_service"
    phase: int = 1
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


# Factory functions pour créer des réponses standardisées
def create_success_response(
    user_id: int,
    message: str,
    intent_result: IntentClassificationResult,
    agent_metrics: AgentMetrics,
    processing_time_ms: int,
    request_id: Optional[str] = None
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
        status=ProcessingStatus.SUCCESS
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