"""
Modèles de réponse Phase 3 - Intentions + Entités + Requêtes search_service
Extension des modèles existants avec génération requêtes
"""
from pydantic import BaseModel, field_validator, ConfigDict, computed_field, field_serializer
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from conversation_service.models.responses.conversation_responses import (
    ConversationResponse, AgentMetrics, ProcessingStatus, IntentClassificationResult
)
from conversation_service.models.contracts.search_service import (
    SearchQuery, QueryValidationResult, QueryGenerationResponse
)


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


class ConversationResponsePhase3(ConversationResponse):
    """Réponse complète Phase 3: Intentions + Entités + Requêtes search_service"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "user_id": 123,
                "message": "Combien j'ai dépensé chez Amazon ce mois ?",
                "timestamp": "2024-08-26T14:30:00Z",
                "processing_time_ms": 876,
                "intent": {
                    "intent_type": "SEARCH_BY_MERCHANT",
                    "confidence": 0.94
                },
                "entities": {
                    "merchants": ["Amazon"],
                    "dates": {"normalized": {"gte": "2024-08-01", "lte": "2024-08-31"}}
                },
                "search_query": {
                    "user_id": 123,
                    "filters": {
                        "merchant_name": {"match": "Amazon"},
                        "date": {"gte": "2024-08-01", "lte": "2024-08-31"},
                        "transaction_type": "debit"
                    },
                    "aggregations": {
                        "merchant_analysis": {
                            "terms": {"field": "merchant_name.keyword", "size": 10},
                            "aggs": {"total_spent": {"sum": {"field": "amount_abs"}}}
                        }
                    },
                    "sort": [{"date": {"order": "desc"}}],
                    "page_size": 20
                },
                "query_validation": {
                    "schema_valid": True,
                    "contract_compliant": True,
                    "estimated_performance": "optimal"
                },
                "phase": 3
            }
        }
    )
    
    # Phase 3 - Requête search_service
    search_query: Optional[SearchQuery] = None
    query_validation: Optional[QueryValidationResult] = None
    query_generation_metrics: Optional[QueryGenerationMetrics] = None
    
    # Métadonnées traitement détaillées
    processing_steps: List[ProcessingSteps] = []
    agent_metrics_detailed: Dict[str, Any] = {}
    
    # Override phase par défaut
    phase: int = 3
    
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
    
    @computed_field
    @property
    def performance_summary_phase3(self) -> Dict[str, Any]:
        """Résumé performance Phase 3"""
        base_summary = self.performance_summary
        
        phase3_summary = {
            **base_summary,
            "phase": 3,
            "query_generation_success": self.query_generation_success,
            "agents_used": len(self.processing_steps),
            "cache_efficiency": self.cache_efficiency,
            "estimated_query_performance": (
                self.query_validation.estimated_performance 
                if self.query_validation else "unknown"
            )
        }
        
        if self.query_generation_metrics:
            phase3_summary.update({
                "generation_confidence": self.query_generation_metrics.generation_confidence,
                "optimizations_applied": self.query_generation_metrics.optimizations_applied,
                "estimated_results": self.query_generation_metrics.estimated_results_count
            })
        
        return phase3_summary
    
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
    
    def to_phase2_response(self) -> ConversationResponse:
        """Conversion vers réponse Phase 2 (sans requête search_service)"""
        return ConversationResponse(
            user_id=self.user_id,
            sub=self.sub,
            message=self.message,
            timestamp=self.timestamp,
            request_id=self.request_id,
            intent=self.intent,
            agent_metrics=self.agent_metrics,
            processing_time_ms=self.processing_time_ms,
            status=self.status,
            phase=2,  # Downgrade vers Phase 2
            warnings=self.warnings,
            debug_info=self.debug_info,
            entities=self.entities
        )
    
    def to_minimal_dict_phase3(self) -> Dict[str, Any]:
        """Version minimale Phase 3 pour logs"""
        base_minimal = self.to_minimal_dict()
        return {
            **base_minimal,
            "phase": 3,
            "has_search_query": self.has_search_query,
            "query_generation_success": self.query_generation_success,
            "agents_used": self.agents_sequence,
            "cache_efficiency": self.cache_efficiency
        }


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


class ConversationResponsePhase3Error(ConversationResponsePhase3):
    """Réponse Phase 3 avec erreur de génération de requête"""
    
    # Erreur spécifique
    query_generation_error: QueryGenerationError
    
    # Override status par défaut
    status: ProcessingStatus = ProcessingStatus.PARTIAL_SUCCESS
    
    @computed_field
    @property
    def has_fallback(self) -> bool:
        """Indique si un fallback a été appliqué"""
        return self.query_generation_error.fallback_applied
    
    def to_error_summary(self) -> Dict[str, Any]:
        """Résumé d'erreur pour monitoring"""
        return {
            "user_id": self.user_id,
            "error_type": self.query_generation_error.error_type,
            "error_agent": self.query_generation_error.error_agent,
            "fallback_applied": self.query_generation_error.fallback_applied,
            "processing_time_ms": self.processing_time_ms,
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat()
        }


# Factory pour création réponses selon résultat
class ConversationResponseFactory:
    """Factory pour créer les bonnes réponses selon le contexte"""
    
    @staticmethod
    def create_phase3_success(
        base_response: ConversationResponse,
        search_query: SearchQuery,
        query_validation: QueryValidationResult,
        processing_steps: List[ProcessingSteps],
        query_metrics: Optional[QueryGenerationMetrics] = None
    ) -> ConversationResponsePhase3:
        """Création réponse Phase 3 réussie"""
        
        response = ConversationResponsePhase3(
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
            processing_steps=processing_steps
        )
        
        if query_metrics:
            response.set_query_generation_metrics(query_metrics)
        
        return response
    
    @staticmethod
    def create_phase3_error(
        base_response: ConversationResponse,
        error: QueryGenerationError,
        processing_steps: List[ProcessingSteps]
    ) -> ConversationResponsePhase3Error:
        """Création réponse Phase 3 avec erreur"""
        
        return ConversationResponsePhase3Error(
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
            query_generation_error=error
        )