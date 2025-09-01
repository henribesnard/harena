"""
Modèles de réponse Phase 4 - Exécution complète avec résultats search_service
Extension Phase 3 avec résultats réels et métriques de résilience
"""
from pydantic import BaseModel, field_validator, ConfigDict, computed_field, field_serializer
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from conversation_service.models.responses.conversation_responses_phase3 import (
    ConversationResponsePhase3, ProcessingSteps, QueryGenerationMetrics
)
from conversation_service.models.contracts.search_service import (
    SearchResponse, QueryValidationResult
)


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


class ConversationResponsePhase4(ConversationResponsePhase3):
    """Réponse complète Phase 4: Tout Phase 3 + Résultats search_service réels"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "user_id": 123,
                "message": "Combien j'ai dépensé chez Amazon ce mois ?",
                "timestamp": "2024-08-26T14:30:00Z",
                "processing_time_ms": 1245,
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
                        "date": {"gte": "2024-08-01", "lte": "2024-08-31"}
                    }
                },
                "search_results": {
                    "hits": [{"_source": {"amount": -45.99, "merchant_name": "Amazon"}}],
                    "total_hits": 12,
                    "aggregations": {"total_spent": {"value": -287.45}}
                },
                "resilience_metrics": {
                    "circuit_breaker_triggered": False,
                    "retry_attempts": 0,
                    "cache_hit": False,
                    "fallback_used": False,
                    "search_execution_time_ms": 234
                },
                "phase": 4
            }
        }
    )
    
    # Phase 4 - Résultats search_service
    search_results: Optional[SearchResponse] = None
    resilience_metrics: Optional[ResilienceMetrics] = None
    search_metrics: Optional[SearchMetrics] = None
    
    # Override phase par défaut
    phase: int = 4
    
    @field_validator('search_results')
    @classmethod
    def validate_search_results(cls, v: Optional[SearchResponse]) -> Optional[SearchResponse]:
        # search_results peut être None si recherche échouée
        return v
    
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
    
    @computed_field
    @property
    def performance_summary_phase4(self) -> Dict[str, Any]:
        """Résumé performance Phase 4 complet"""
        base_summary = self.performance_summary_phase3
        
        phase4_summary = {
            **base_summary,
            "phase": 4,
            "search_execution_success": self.search_execution_success,
            "has_search_results": self.has_search_results,
            "total_time_with_search_ms": self.total_processing_time_with_search_ms
        }
        
        # Ajouter métriques de résilience
        if self.resilience_metrics:
            phase4_summary.update({
                "circuit_breaker_triggered": self.resilience_metrics.circuit_breaker_triggered,
                "retry_attempts": self.resilience_metrics.retry_attempts,
                "cache_hit": self.resilience_metrics.cache_hit,
                "fallback_used": self.resilience_metrics.fallback_used,
                "resilience_overhead_ms": self.resilience_metrics.total_resilience_overhead_ms
            })
        
        # Ajouter métriques search
        if self.search_metrics:
            phase4_summary.update({
                "total_hits": self.search_metrics.total_hits,
                "results_relevance": self.search_metrics.results_relevance,
                "search_service_time_ms": self.search_metrics.search_service_took_ms,
                "estimated_completeness": self.search_metrics.estimated_completeness
            })
        
        return phase4_summary
    
    def set_resilience_metrics(self, metrics: ResilienceMetrics) -> None:
        """Définit les métriques de résilience"""
        self.resilience_metrics = metrics
    
    def set_search_metrics(self, metrics: SearchMetrics) -> None:
        """Définit les métriques search"""
        self.search_metrics = metrics
    
    def to_phase3_response(self) -> ConversationResponsePhase3:
        """Conversion vers réponse Phase 3 (sans résultats search)"""
        return ConversationResponsePhase3(
            user_id=self.user_id,
            sub=self.sub,
            message=self.message,
            timestamp=self.timestamp,
            request_id=self.request_id,
            intent=self.intent,
            agent_metrics=self.agent_metrics,
            processing_time_ms=self.processing_time_ms,
            status=self.status,
            warnings=self.warnings,
            debug_info=self.debug_info,
            entities=self.entities,
            search_query=self.search_query,
            query_validation=self.query_validation,
            query_generation_metrics=self.query_generation_metrics,
            processing_steps=self.processing_steps,
            agent_metrics_detailed=self.agent_metrics_detailed
        )
    
    def to_minimal_dict_phase4(self) -> Dict[str, Any]:
        """Version minimale Phase 4 pour logs"""
        base_minimal = self.to_minimal_dict_phase3()
        return {
            **base_minimal,
            "phase": 4,
            "has_search_results": self.has_search_results,
            "search_execution_success": self.search_execution_success,
            "results_count": self.search_results.total_hits if self.search_results else 0,
            "fallback_used": self.resilience_metrics.fallback_used if self.resilience_metrics else False
        }


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


class ConversationResponsePhase4Error(ConversationResponsePhase4):
    """Réponse Phase 4 avec erreur d'exécution search"""
    
    # Erreur spécifique search
    search_execution_error: SearchExecutionError
    
    # Override status par défaut
    status: str = "partial_success"  # Peut avoir intent/entities mais pas résultats
    
    @computed_field
    @property
    def has_recovery_attempt(self) -> bool:
        """Indique si une récupération a été tentée"""
        return self.search_execution_error.recovery_attempted
    
    @computed_field
    @property
    def recovery_successful(self) -> bool:
        """Indique si la récupération a réussi"""
        return (self.search_execution_error.recovery_attempted and 
                self.search_execution_error.recovery_successful)
    
    def to_error_summary(self) -> Dict[str, Any]:
        """Résumé d'erreur pour monitoring Phase 4"""
        return {
            "user_id": self.user_id,
            "phase": self.phase,
            "error_type": self.search_execution_error.error_type,
            "error_component": self.search_execution_error.error_component,
            "recovery_attempted": self.search_execution_error.recovery_attempted,
            "recovery_successful": self.search_execution_error.recovery_successful,
            "circuit_breaker_state": self.search_execution_error.circuit_breaker_state,
            "retry_attempts": self.search_execution_error.retry_attempts_made,
            "processing_time_ms": self.processing_time_ms,
            "has_fallback_results": self.has_search_results,
            "timestamp": self.timestamp.isoformat()
        }


# Factory étendue pour Phase 4
class ConversationResponseFactoryPhase4:
    """Factory pour créer les réponses Phase 4"""
    
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
    def create_phase4_success(
        base_response: ConversationResponsePhase3,
        search_results: SearchResponse,
        resilience_metrics: ResilienceMetrics,
        search_metrics: Optional[SearchMetrics] = None
    ) -> ConversationResponsePhase4:
        """Création réponse Phase 4 réussie avec résultats"""
        
        response = ConversationResponsePhase4(
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
            resilience_metrics=resilience_metrics
        )
        
        if search_metrics:
            response.set_search_metrics(search_metrics)
        
        return response
    
    @staticmethod
    def create_phase4_error(
        base_response: ConversationResponsePhase3,
        error: SearchExecutionError,
        resilience_metrics: Optional[ResilienceMetrics] = None,
        partial_results: Optional[SearchResponse] = None
    ) -> ConversationResponsePhase4Error:
        """Création réponse Phase 4 avec erreur search"""
        
        response = ConversationResponsePhase4Error(
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
            search_execution_error=error
        )
        
        # Ajouter résultats partiels si disponibles (fallback)
        if partial_results:
            response.search_results = partial_results
        
        # Ajouter métriques résilience si disponibles
        if resilience_metrics:
            response.set_resilience_metrics(resilience_metrics)
        
        return response
    
    @staticmethod
    def create_from_search_executor_response(
        base_response: ConversationResponsePhase3,
        executor_response,  # SearchExecutorResponse
        processing_step: ProcessingSteps
    ) -> Union[ConversationResponsePhase4, ConversationResponsePhase4Error]:
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
        base_with_steps = ConversationResponsePhase3(
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
            agent_metrics_detailed=base_response.agent_metrics_detailed
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
                results_relevance=ConversationResponseFactoryPhase4._map_performance_to_relevance(executor_response.estimated_performance)
            )
            
            return ConversationResponseFactoryPhase4.create_phase4_success(
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
            
            return ConversationResponseFactoryPhase4.create_phase4_error(
                base_with_steps,
                error,
                resilience_metrics,
                executor_response.search_results  # Peut être None ou résultats fallback
            )