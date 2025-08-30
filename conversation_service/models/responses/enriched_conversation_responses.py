"""
Modèles de réponses enrichis pour intégration AutoGen
Compatible avec ConversationResponse existant + métadonnées équipe multi-agents
"""

from pydantic import BaseModel, field_validator, ConfigDict, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from enum import Enum

# Import modèles existants (compatibilité totale)
from .conversation_responses import (
    ConversationResponse,
    IntentClassificationResult,
    IntentConfidenceLevel,
    ProcessingStatus,
    CacheStatus,
    AgentMetrics
)


class ProcessingMode(str, Enum):
    """Modes de traitement disponibles"""
    SINGLE_AGENT = "single_agent"              # Mode existant classique
    MULTI_AGENT_TEAM = "multi_agent_team"      # Mode AutoGen équipe
    FALLBACK_CHAIN = "fallback_chain"          # AutoGen puis fallback existant


class TeamWorkflowStatus(str, Enum):
    """Statuts workflow équipe AutoGen"""
    COMPLETED = "completed"                     # Workflow complet (Intent + Entity)
    PARTIAL = "partial"                         # Workflow partiel (Intent uniquement)
    FAILED = "failed"                          # Workflow échoué
    TIMEOUT = "timeout"                        # Timeout workflow
    FALLBACK_APPLIED = "fallback_applied"      # Fallback vers agents individuels


class EntityExtractionResult(BaseModel):
    """Résultat extraction entités (nouveau avec AutoGen)"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: Optional[str] = None
    extraction_strategy: Optional[str] = None  # "focused", "comprehensive", "minimal"
    entities_count: int = Field(ge=0, default=0)
    processing_time_ms: Optional[int] = Field(ge=0, default=None)
    
    @field_validator('entities_count', mode='before')
    @classmethod
    def compute_entities_count(cls, v, info):
        """Calcule automatiquement le nombre d'entités"""
        if 'entities' in info.data:
            entities_dict = info.data['entities']
            return sum(len(entity_list) for entity_list in entities_dict.values() if isinstance(entity_list, list))
        return v or 0


class CoherenceValidation(BaseModel):
    """Validation cohérence intention-entités"""
    model_config = ConfigDict(validate_assignment=True)
    
    score: float = Field(ge=0.0, le=1.0)
    threshold_met: bool
    validation_rules_applied: List[str] = Field(default_factory=list)
    coherence_issues: Optional[List[str]] = None
    
    @field_validator('threshold_met', mode='before')
    @classmethod
    def compute_threshold_met(cls, v, info):
        """Calcule automatiquement si seuil atteint"""
        if 'score' in info.data:
            return info.data['score'] > 0.7  # Seuil cohérence
        return v or False


class AutoGenTeamMetadata(BaseModel):
    """Métadonnées spécifiques équipe AutoGen"""
    model_config = ConfigDict(validate_assignment=True)
    
    # Workflow
    processing_mode: ProcessingMode
    team_name: str
    workflow_status: TeamWorkflowStatus
    agents_involved: List[str] = Field(default_factory=list)
    
    # Performance
    total_processing_time_ms: int = Field(ge=0)
    intent_processing_time_ms: Optional[int] = Field(ge=0, default=None)
    entity_processing_time_ms: Optional[int] = Field(ge=0, default=None)
    
    # Qualité
    coherence_validation: CoherenceValidation
    workflow_success: bool
    
    # Cache et Infrastructure
    from_team_cache: bool = False
    cache_strategy: Optional[str] = None
    
    # Fallback
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    
    # Debug (optionnel)
    conversation_history: Optional[List[Dict]] = None
    agent_errors: Optional[Dict[str, str]] = None


class EnrichedConversationResponse(ConversationResponse):
    """
    Réponse enrichie compatible avec ConversationResponse existant
    Ajoute métadonnées AutoGen sans casser compatibilité
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                # Compatibilité existant (champs identiques)
                "user_id": 123,
                "sub": 123,
                "message": "Combien j'ai dépensé chez Carrefour ce mois ?",
                "timestamp": "2024-08-26T14:30:00+00:00",
                "processing_time_ms": 1850,
                "status": "success",
                "intent": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "confidence": 0.92,
                    "reasoning": "Recherche dépenses chez marchand spécifique",
                    "original_message": "Combien j'ai dépensé chez Carrefour ce mois ?",
                    "category": "FINANCIAL_QUERY",
                    "is_supported": True,
                    "alternatives": [],
                    "processing_time_ms": 800
                },
                "agent_metrics": {
                    "agent_used": "multi_agent_team",
                    "model_used": "deepseek-chat",
                    "tokens_consumed": 280,
                    "processing_time_ms": 1850,
                    "confidence_threshold_met": True,
                    "cache_hit": False
                },
                
                # Extensions AutoGen (nouveaux champs)
                "entities_extraction": {
                    "entities": {
                        "merchants": ["Carrefour"],
                        "dates": [{"type": "period", "value": "2024-01"}]
                    },
                    "confidence": 0.88,
                    "reasoning": "Extraction focalisée selon intention SEARCH_BY_MERCHANT",
                    "entities_count": 2,
                    "processing_time_ms": 950
                },
                "autogen_metadata": {
                    "processing_mode": "multi_agent_team",
                    "team_name": "multi_agent_financial_team",
                    "workflow_status": "completed",
                    "agents_involved": ["intent_classifier", "entity_extractor"],
                    "total_processing_time_ms": 1850,
                    "coherence_validation": {
                        "score": 0.85,
                        "threshold_met": True
                    },
                    "workflow_success": True
                }
            }
        }
    )
    
    # Extensions AutoGen (champs additionnels optionnels)
    entities_extraction: Optional[EntityExtractionResult] = None
    autogen_metadata: Optional[AutoGenTeamMetadata] = None
    
    @classmethod
    def from_team_results(
        cls,
        user_id: int,
        message: str, 
        team_results: Dict[str, Any],
        processing_time_ms: int,
        sub: Optional[int] = None
    ) -> "EnrichedConversationResponse":
        """
        Factory method pour créer réponse enrichie depuis résultats équipe AutoGen
        Maintient compatibilité totale avec ConversationResponse existant
        """
        
        intent_result = team_results.get("intent_result", {})
        entities_result = team_results.get("entities_result", {})
        coherence_validation = team_results.get("coherence_validation", {})
        
        # Intent Response (format existant identique)
        intent_response = IntentClassificationResult(
            intent_type=intent_result.get("intent", "UNCLEAR_INTENT"),
            confidence=intent_result.get("confidence", 0.0),
            reasoning=intent_result.get("reasoning", ""),
            original_message=message,
            category="FINANCIAL_QUERY",
            is_supported=True,
            alternatives=[],
            processing_time_ms=processing_time_ms // 2  # Estimation intent
        )
        
        # Agent Metrics (format existant adapté)
        agent_metrics = AgentMetrics(
            agent_used="multi_agent_team",
            model_used="deepseek-chat", 
            tokens_consumed=250,  # Estimation
            processing_time_ms=processing_time_ms,
            confidence_threshold_met=intent_result.get("confidence", 0.0) > 0.7,
            cache_hit=team_results.get("from_cache", False),
            retry_count=0,
            error_count=0
        )
        
        # Entity Extraction (nouveau)
        entities_extraction = EntityExtractionResult(
            entities=entities_result.get("entities", {}),
            confidence=entities_result.get("confidence", 0.0),
            reasoning=entities_result.get("reasoning", ""),
            extraction_strategy=entities_result.get("team_context", {}).get("extraction_strategy_used", "focused"),
            processing_time_ms=processing_time_ms // 2  # Estimation entity
        )
        
        # Coherence Validation
        coherence = CoherenceValidation(
            score=coherence_validation.get("score", 0.0),
            threshold_met=coherence_validation.get("threshold_met", False),
            validation_rules_applied=["intention_entity_coherence"]
        )
        
        # AutoGen Metadata
        autogen_metadata = AutoGenTeamMetadata(
            processing_mode=ProcessingMode.MULTI_AGENT_TEAM,
            team_name="multi_agent_financial_team",
            workflow_status=TeamWorkflowStatus.COMPLETED if team_results.get("workflow_success") else TeamWorkflowStatus.PARTIAL,
            agents_involved=team_results.get("agents_sequence", []),
            total_processing_time_ms=processing_time_ms,
            coherence_validation=coherence,
            workflow_success=team_results.get("workflow_success", False),
            from_team_cache=team_results.get("from_cache", False),
            fallback_applied=team_results.get("error_context", {}).get("fallback_strategy") is not None
        )
        
        # Construction réponse enrichie (compatible existant)
        return cls(
            user_id=user_id,
            sub=sub or user_id,
            message=message,
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms,
            status=ProcessingStatus.SUCCESS if team_results.get("workflow_success") else ProcessingStatus.PARTIAL_SUCCESS,
            intent=intent_response,
            agent_metrics=agent_metrics,
            
            # Extensions AutoGen
            entities_extraction=entities_extraction,
            autogen_metadata=autogen_metadata
        )
    
    @classmethod 
    def from_fallback_single_agent(
        cls,
        base_response: ConversationResponse,
        fallback_reason: str
    ) -> "EnrichedConversationResponse":
        """
        Factory method pour créer réponse enrichie depuis fallback agent unique
        Préserve réponse existante + ajoute contexte fallback
        """
        
        # Métadonnées fallback
        autogen_metadata = AutoGenTeamMetadata(
            processing_mode=ProcessingMode.FALLBACK_CHAIN,
            team_name="fallback_single_agent",
            workflow_status=TeamWorkflowStatus.FALLBACK_APPLIED,
            agents_involved=[base_response.agent_metrics.agent_used],
            total_processing_time_ms=base_response.processing_time_ms,
            coherence_validation=CoherenceValidation(
                score=0.5,  # Score neutre pour fallback
                threshold_met=False
            ),
            workflow_success=base_response.status == ProcessingStatus.SUCCESS,
            fallback_applied=True,
            fallback_reason=fallback_reason
        )
        
        # Copy base response + add AutoGen metadata
        return cls(
            **base_response.model_dump(),
            autogen_metadata=autogen_metadata
        )


class TeamHealthResponse(BaseModel):
    """Réponse health check équipe AutoGen"""
    model_config = ConfigDict(validate_assignment=True)
    
    service: str = "conversation_service"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Disponibilité modes
    single_agent_available: bool
    multi_agent_team_available: bool
    current_processing_mode: ProcessingMode
    
    # Détails AutoGen (si disponible)
    autogen_details: Optional[Dict[str, Any]] = None
    
    # Performance récente
    recent_performance: Optional[Dict[str, float]] = None


class TeamMetricsResponse(BaseModel):
    """Réponse métriques équipe AutoGen détaillées"""
    model_config = ConfigDict(validate_assignment=True)
    
    available: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Statistiques équipe
    statistics: Optional[Dict[str, Any]] = None
    
    # Health status
    health: Optional[Dict[str, Any]] = None
    
    # Métriques comparatives
    performance_comparison: Optional[Dict[str, Any]] = None