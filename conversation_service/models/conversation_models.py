"""
💬 Modèles Pydantic pour conversation service - PHASE 1 (L0 Pattern Matching)

Modèles de données avec évitement des imports circulaires,
validation robuste et sérialization optimisée.
Focus Phase 1 : Détails précis pour pattern matching L0.
"""

from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import time

# ✅ Pas d'imports de modules conversation_service pour éviter circuits
# Tous les modèles sont définis dans ce module uniquement

# ==========================================
# ENUMS ET TYPES DE BASE
# ==========================================

class FinancialIntent(str, Enum):
    """Types d'intentions financières supportées - Focus Phase 1"""
    
    # ========== INTENTIONS L0 PRINCIPALES ==========
    BALANCE_CHECK = "BALANCE_CHECK"
    EXPENSE_ANALYSIS = "EXPENSE_ANALYSIS"
    TRANSFER = "TRANSFER"
    CARD_MANAGEMENT = "CARD_MANAGEMENT"
    
    # ========== INTENTIONS L0 SECONDAIRES ==========
    BILL_PAYMENT = "BILL_PAYMENT"
    TRANSACTION_HISTORY = "TRANSACTION_HISTORY"
    ACCOUNT_MANAGEMENT = "ACCOUNT_MANAGEMENT"
    
    # ========== INTENTIONS SYSTÈME ==========
    GREETING = "GREETING"
    HELP = "HELP"
    GOODBYE = "GOODBYE"
    
    # ========== INTENTIONS FUTURES (L1/L2) ==========
    INVESTMENT_QUERY = "INVESTMENT_QUERY"
    LOAN_INQUIRY = "LOAN_INQUIRY"
    BUDGET_PLANNING = "BUDGET_PLANNING"
    SAVINGS_GOAL = "SAVINGS_GOAL"
    FINANCIAL_ADVICE = "FINANCIAL_ADVICE"
    
    # ========== FALLBACKS ==========
    UNKNOWN = "UNKNOWN"
    SYSTEM_ERROR = "SYSTEM_ERROR"

class ProcessingLevel(str, Enum):
    """Niveaux de traitement du pipeline - Phase 1 focus L0"""
    L0_PATTERN = "L0_PATTERN"                    # ✅ Phase 1 - Pattern matching
    L1_LIGHTWEIGHT = "L1_LIGHTWEIGHT"           # Phase 2 future
    L2_LLM = "L2_LLM"                          # Phase 3 future
    ERROR_FALLBACK = "ERROR_FALLBACK"
    ERROR_TIMEOUT = "ERROR_TIMEOUT"
    ERROR_VALIDATION = "ERROR_VALIDATION"
    SYSTEM_ERROR = "SYSTEM_ERROR"               # ✅ Phase 1 - Erreurs système

class ConfidenceLevel(str, Enum):
    """Niveaux de confiance"""
    VERY_HIGH = "VERY_HIGH"    # > 0.9
    HIGH = "HIGH"              # 0.8 - 0.9
    MEDIUM = "MEDIUM"          # 0.6 - 0.8
    LOW = "LOW"                # 0.4 - 0.6
    VERY_LOW = "VERY_LOW"      # < 0.4

class PatternType(str, Enum):
    """Types de patterns L0 - Phase 1"""
    DIRECT_KEYWORD = "DIRECT_KEYWORD"           # "solde", "virement"
    QUESTION_PHRASE = "QUESTION_PHRASE"         # "quel est mon solde"
    ACTION_VERB = "ACTION_VERB"                 # "faire un virement"
    AMOUNT_EXTRACTION = "AMOUNT_EXTRACTION"     # "virer 100€"
    CATEGORY_SPECIFIC = "CATEGORY_SPECIFIC"     # "dépenses restaurant"
    TEMPORAL_CONTEXT = "TEMPORAL_CONTEXT"      # "ce mois", "aujourd'hui"
    GREETING_SYSTEM = "GREETING_SYSTEM"        # "bonjour", "aide"

# ==========================================
# MODÈLES DE MÉTADONNÉES ENRICHIES PHASE 1
# ==========================================

class ProcessingMetadata(BaseModel):
    """Métadonnées de traitement enrichies - Phase 1"""
    request_id: str
    level_used: str
    processing_time_ms: float
    cache_hit: bool = False
    engine_latency_ms: Optional[float] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    error: Optional[str] = None
    
    # ✅ NOUVEAUX CHAMPS PHASE 1 - Détails Pattern Matching
    pattern_matched: Optional[str] = Field(None, description="Nom du pattern L0 utilisé")
    pattern_type: Optional[str] = Field(None, description="Type de pattern (DIRECT_KEYWORD, etc.)")
    confidence_reasoning: Optional[str] = Field(None, description="Explication du score confiance")
    matched_text: Optional[str] = Field(None, description="Texte exact qui a matché")
    matched_position: Optional[Dict[str, int]] = Field(None, description="Position du match dans le texte")
    entities_extracted: Optional[int] = Field(0, description="Nombre d'entités extraites")
    fallback_reason: Optional[str] = Field(None, description="Raison du fallback si applicable")
    
    # ✅ MÉTRIQUES L0 SPÉCIALISÉES
    pattern_confidence_base: Optional[float] = Field(None, description="Confiance pattern avant ajustements")
    text_normalization_applied: Optional[bool] = Field(None, description="Normalisation texte appliquée")
    cache_key_used: Optional[str] = Field(None, description="Clé cache utilisée")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "level_used": "L0_PATTERN",
                "processing_time_ms": 8.5,
                "cache_hit": False,
                "engine_latency_ms": 7.2,
                "timestamp": 1640995200,
                "pattern_matched": "direct_balance",
                "pattern_type": "DIRECT_KEYWORD",
                "confidence_reasoning": "Correspondance exacte mot-clé 'solde' avec boost confiance",
                "matched_text": "solde",
                "matched_position": {"start": 0, "end": 5},
                "entities_extracted": 0,
                "pattern_confidence_base": 0.90,
                "text_normalization_applied": True,
                "cache_key_used": "l0_pattern_5f8a7b2c"
            }
        }

class ConfidenceScore(BaseModel):
    """Score de confiance avec niveau et détails Phase 1"""
    score: float = Field(..., ge=0.0, le=1.0)
    level: ConfidenceLevel
    reasoning: str = Field(..., description="Explication détaillée du score")
    
    # ✅ NOUVEAUX CHAMPS PHASE 1
    base_score: Optional[float] = Field(None, description="Score avant ajustements")
    adjustments: Optional[Dict[str, float]] = Field(None, description="Ajustements appliqués")
    
    @validator('level', always=True)
    def determine_confidence_level(cls, v, values):
        """Détermine automatiquement le niveau selon le score"""
        score = values.get('score', 0.0)
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.95,
                "level": "VERY_HIGH",
                "reasoning": "Pattern direct 'solde' avec contexte bancaire",
                "base_score": 0.90,
                "adjustments": {"keyword_boost": 0.05}
            }
        }

# ==========================================
# MODÈLES DE REQUÊTE ET RÉPONSE
# ==========================================

class ChatRequest(BaseModel):
    """Requête de chat utilisateur - Phase 1"""
    message: str = Field(..., min_length=1, max_length=2000, description="Message utilisateur")
    user_id: int = Field(..., gt=0, description="ID utilisateur")
    conversation_id: Optional[str] = Field(None, max_length=100, description="ID conversation optionnel")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contexte additionnel")
    
    # ✅ NOUVEAUX CHAMPS PHASE 1
    force_level: Optional[str] = Field(None, description="Forcer un niveau spécifique (debug)")
    enable_cache: Optional[bool] = Field(True, description="Activer cache patterns")
    debug_mode: Optional[bool] = Field(False, description="Mode debug avec détails")
    
    @validator('message')
    def validate_message(cls, v):
        """Validation et nettoyage du message"""
        if not v or not v.strip():
            raise ValueError("Message ne peut pas être vide")
        return v.strip()
    
    @validator('context')
    def validate_context(cls, v):
        """Validation du contexte"""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("Context doit être un dictionnaire")
        return v
    
    @validator('force_level')
    def validate_force_level(cls, v):
        """Validation niveau forcé"""
        if v is not None and v not in ["L0", "L1", "L2"]:
            raise ValueError("force_level doit être L0, L1 ou L2")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Quel est mon solde ?",
                "user_id": 123,
                "conversation_id": "conv_456",
                "context": {"channel": "web"},
                "force_level": None,
                "enable_cache": True,
                "debug_mode": False
            }
        }

class ChatResponse(BaseModel):
    """Réponse du service conversation - Phase 1 enrichie"""
    request_id: str = Field(..., description="ID unique de la requête")
    intent: str = Field(..., description="Intention détectée")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score de confiance")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Entités extraites")
    message: Optional[str] = Field(None, description="Message de réponse")
    suggested_actions: Optional[List[str]] = Field(None, description="Actions suggérées")
    processing_metadata: ProcessingMetadata = Field(..., description="Métadonnées de traitement")
    success: bool = Field(True, description="Succès du traitement")
    error: Optional[str] = Field(None, description="Message d'erreur si échec")
    
    # ✅ NOUVEAUX CHAMPS PHASE 1
    confidence_details: Optional[ConfidenceScore] = Field(None, description="Détails score confiance")
    pattern_analysis: Optional[Dict[str, Any]] = Field(None, description="Analyse pattern pour debug")
    alternatives: Optional[List[Dict[str, Any]]] = Field(None, description="Intentions alternatives")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "intent": "BALANCE_CHECK",
                "confidence": 0.95,
                "entities": {"account_type": "checking"},
                "message": "Je vais vérifier votre solde",
                "suggested_actions": ["show_balance", "show_transactions"],
                "success": True,
                "confidence_details": {
                    "score": 0.95,
                    "level": "VERY_HIGH",
                    "reasoning": "Pattern direct 'solde' avec contexte bancaire"
                },
                "pattern_analysis": {
                    "patterns_tested": 5,
                    "best_match": "direct_balance",
                    "match_strength": "perfect"
                },
                "processing_metadata": {
                    "request_id": "req_1234567890",
                    "level_used": "L0_PATTERN",
                    "processing_time_ms": 8.5,
                    "pattern_matched": "direct_balance"
                }
            }
        }

# ==========================================
# MODÈLES D'ENTITÉS FINANCIÈRES
# ==========================================

class FinancialEntity(BaseModel):
    """Entité financière extraite - Phase 1"""
    type: str = Field(..., description="Type d'entité (amount, date, account, etc.)")
    value: Union[str, int, float] = Field(..., description="Valeur de l'entité")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance extraction")
    position: Optional[Dict[str, int]] = Field(None, description="Position dans le texte")
    
    # ✅ NOUVEAUX CHAMPS PHASE 1
    extraction_method: Optional[str] = Field(None, description="Méthode extraction (regex, pattern)")
    normalized_value: Optional[Union[str, int, float]] = Field(None, description="Valeur normalisée")
    currency: Optional[str] = Field(None, description="Devise si applicable")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "amount",
                "value": "150.50€",
                "confidence": 0.9,
                "position": {"start": 10, "end": 17},
                "extraction_method": "regex_currency",
                "normalized_value": 150.50,
                "currency": "EUR"
            }
        }

class PatternMatch(BaseModel):
    """Détails d'un match de pattern L0"""
    pattern_name: str = Field(..., description="Nom du pattern")
    pattern_type: PatternType = Field(..., description="Type de pattern")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance du match")
    matched_text: str = Field(..., description="Texte qui a matché")
    position: Dict[str, int] = Field(..., description="Position dans le texte original")
    entities: List[FinancialEntity] = Field(default_factory=list, description="Entités extraites")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pattern_name": "transfer_with_amount",
                "pattern_type": "AMOUNT_EXTRACTION",
                "confidence": 0.92,
                "matched_text": "virer 100€",
                "position": {"start": 0, "end": 9},
                "entities": [
                    {
                        "type": "amount",
                        "value": "100€",
                        "confidence": 0.95,
                        "normalized_value": 100.0,
                        "currency": "EUR"
                    }
                ]
            }
        }

# ==========================================
# MODÈLES DE MÉTRIQUES PHASE 1
# ==========================================

class L0PerformanceMetrics(BaseModel):
    """Métriques spécialisées L0 Pattern Matching"""
    total_requests: int = 0
    l0_successful_requests: int = 0
    l0_failed_requests: int = 0
    avg_l0_latency_ms: float = 0.0
    
    # Distribution patterns
    pattern_usage: Dict[str, int] = Field(default_factory=dict)
    pattern_success_rate: Dict[str, float] = Field(default_factory=dict)
    pattern_avg_latency: Dict[str, float] = Field(default_factory=dict)
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    avg_cache_lookup_ms: float = 0.0
    
    # Confiance distribution
    confidence_distribution: Dict[str, int] = Field(default_factory=dict)
    
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    
    @property
    def l0_success_rate(self) -> float:
        """Taux de succès L0"""
        total = self.l0_successful_requests + self.l0_failed_requests
        return (self.l0_successful_requests / total) if total > 0 else 0.0
    
    @property
    def target_l0_usage_percent(self) -> float:
        """Pourcentage d'usage L0 (target: 85%)"""
        return (self.l0_successful_requests / max(self.total_requests, 1)) * 100
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_requests": 1000,
                "l0_successful_requests": 850,
                "l0_failed_requests": 150,
                "avg_l0_latency_ms": 8.5,
                "pattern_usage": {
                    "direct_balance": 300,
                    "transfer_amount": 200,
                    "expense_category": 150
                },
                "cache_hit_rate": 0.25,
                "confidence_distribution": {
                    "VERY_HIGH": 600,
                    "HIGH": 200,
                    "MEDIUM": 50
                }
            }
        }

class ServiceHealth(BaseModel):
    """État de santé du service - Phase 1"""
    status: str = Field(..., description="healthy, degraded, unhealthy")
    service_name: str = "conversation_service"
    version: str = "1.0.0-phase1"
    phase: str = "L0_PATTERN_MATCHING"
    uptime_seconds: Optional[int] = None
    dependencies: Dict[str, str] = Field(default_factory=dict)
    l0_performance: Optional[L0PerformanceMetrics] = None
    last_check: int = Field(default_factory=lambda: int(time.time()))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    # ✅ MÉTRIQUES PHASE 1 SPÉCIALISÉES
    pattern_matcher_status: Optional[str] = Field(None, description="État du pattern matcher")
    patterns_loaded: Optional[int] = Field(None, description="Nombre de patterns chargés")
    cache_status: Optional[str] = Field(None, description="État du cache")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service_name": "conversation_service",
                "version": "1.0.0-phase1",
                "phase": "L0_PATTERN_MATCHING",
                "pattern_matcher_status": "loaded",
                "patterns_loaded": 65,
                "cache_status": "available",
                "dependencies": {
                    "cache_local": "healthy"
                },
                "last_check": 1640995200,
                "warnings": [],
                "errors": []
            }
        }

# ==========================================
# MODÈLES D'ERREUR SPÉCIALISÉS PHASE 1
# ==========================================

class ConversationError(BaseModel):
    """Erreur structurée du service - Phase 1"""
    error_type: str = Field(..., description="Type d'erreur")
    error_code: str = Field(..., description="Code d'erreur")
    message: str = Field(..., description="Message d'erreur")
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    
    # ✅ ERREURS SPÉCIFIQUES PHASE 1
    pattern_analysis: Optional[Dict[str, Any]] = Field(None, description="Analyse patterns pour debug")
    suggested_fix: Optional[str] = Field(None, description="Suggestion de correction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_type": "pattern_matching_failed",
                "error_code": "L0_NO_MATCH",
                "message": "Aucun pattern ne correspond au message",
                "details": {"patterns_tested": 65, "best_score": 0.3},
                "pattern_analysis": {
                    "message_normalized": "bonjour comment allez vous",
                    "closest_patterns": ["greeting_simple", "help_request"]
                },
                "suggested_fix": "Ajouter patterns pour salutations complexes",
                "request_id": "req_123",
                "timestamp": 1640995200
            }
        }

# ==========================================
# HELPER FUNCTIONS PHASE 1
# ==========================================

def create_l0_error_response(
    request_id: str,
    error_type: str,
    message: str,
    processing_time_ms: float = 0.0,
    pattern_analysis: Dict[str, Any] = None
) -> ChatResponse:
    """Helper pour créer une réponse d'erreur L0"""
    return ChatResponse(
        request_id=request_id,
        intent="UNKNOWN",
        confidence=0.0,
        entities={},
        success=False,
        error=message,
        pattern_analysis=pattern_analysis,
        processing_metadata=ProcessingMetadata(
            request_id=request_id,
            level_used="ERROR_FALLBACK",
            processing_time_ms=processing_time_ms,
            cache_hit=False,
            error=error_type,
            fallback_reason=f"L0 pattern matching failed: {error_type}"
        )
    )

def create_l0_success_response(
    request_id: str,
    intent: str,
    confidence: float,
    entities: Dict[str, Any],
    processing_time_ms: float,
    pattern_match: PatternMatch,
    cache_hit: bool = False,
    message: str = None,
    suggested_actions: List[str] = None,
    confidence_reasoning: str = None
) -> ChatResponse:
    """Helper spécialisé pour réponses L0 avec détails pattern"""
    
    # Construction confidence détaillé
    confidence_details = ConfidenceScore(
        score=confidence,
        reasoning=confidence_reasoning or f"Pattern '{pattern_match.pattern_name}' matched with high confidence",
        base_score=pattern_match.confidence
    )
    
    # Analyse pattern pour debug
    pattern_analysis = {
        "pattern_name": pattern_match.pattern_name,
        "pattern_type": pattern_match.pattern_type.value,
        "matched_text": pattern_match.matched_text,
        "match_position": pattern_match.position,
        "entities_found": len(pattern_match.entities)
    }
    
    return ChatResponse(
        request_id=request_id,
        intent=intent,
        confidence=confidence,
        entities=entities,
        message=message,
        suggested_actions=suggested_actions,
        success=True,
        confidence_details=confidence_details,
        pattern_analysis=pattern_analysis,
        processing_metadata=ProcessingMetadata(
            request_id=request_id,
            level_used="L0_PATTERN",
            processing_time_ms=processing_time_ms,
            cache_hit=cache_hit,
            engine_latency_ms=processing_time_ms,
            pattern_matched=pattern_match.pattern_name,
            pattern_type=pattern_match.pattern_type.value,
            confidence_reasoning=confidence_reasoning,
            matched_text=pattern_match.matched_text,
            matched_position=pattern_match.position,
            entities_extracted=len(entities),
            pattern_confidence_base=pattern_match.confidence,
            text_normalization_applied=True
        )
    )

def create_system_error_response(
    request_id: str,
    error: Exception,
    processing_time_ms: float = 0.0
) -> ChatResponse:
    """Helper pour créer réponse erreur système"""
    return ChatResponse(
        request_id=request_id,
        intent="SYSTEM_ERROR",
        confidence=0.0,
        entities={},
        success=False,
        error=f"Erreur système: {str(error)}",
        processing_metadata=ProcessingMetadata(
            request_id=request_id,
            level_used="SYSTEM_ERROR",
            processing_time_ms=processing_time_ms,
            cache_hit=False,
            error=f"system_error: {type(error).__name__}",
            fallback_reason=f"System exception: {str(error)}"
        )
    )

# ==========================================
# VALIDATION HELPERS PHASE 1
# ==========================================

def validate_l0_targets(metrics: L0PerformanceMetrics) -> Dict[str, bool]:
    """Validation des targets Phase 1"""
    return {
        "latency_target_met": metrics.avg_l0_latency_ms < 10.0,  # <10ms
        "usage_target_met": metrics.target_l0_usage_percent >= 80.0,  # >80%
        "success_rate_met": metrics.l0_success_rate >= 0.85,  # >85%
        "cache_performance_good": metrics.cache_hit_rate >= 0.15  # >15%
    }

def get_pattern_recommendations(metrics: L0PerformanceMetrics) -> List[str]:
    """Recommandations d'amélioration patterns"""
    recommendations = []
    
    if metrics.avg_l0_latency_ms > 10.0:
        recommendations.append("Optimiser patterns les plus lents")
    
    if metrics.target_l0_usage_percent < 80.0:
        recommendations.append("Ajouter patterns pour cas non couverts")
    
    if metrics.cache_hit_rate < 0.15:
        recommendations.append("Améliorer stratégie de cache")
    
    return recommendations

# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    # Enums
    "FinancialIntent", "ProcessingLevel", "ConfidenceLevel", "PatternType",
    
    # Modèles principaux
    "ChatRequest", "ChatResponse", "ProcessingMetadata", "ConfidenceScore",
    
    # Modèles spécialisés
    "FinancialEntity", "PatternMatch", "L0PerformanceMetrics", "ServiceHealth",
    
    # Helpers
    "create_l0_success_response", "create_l0_error_response", "create_system_error_response",
    
    # Validation
    "validate_l0_targets", "get_pattern_recommendations"
]