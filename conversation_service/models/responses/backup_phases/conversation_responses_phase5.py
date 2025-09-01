"""
Modèles de réponse Phase 5 - Génération complète avec réponses naturelles
Extension Phase 4 avec génération de réponses contextualisées et insights automatiques
"""
from pydantic import BaseModel, field_validator, ConfigDict, computed_field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

from conversation_service.models.responses.conversation_responses_phase4 import (
    ConversationResponsePhase4, ResilienceMetrics, SearchMetrics
)


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


class ConversationResponsePhase5(ConversationResponsePhase4):
    """Réponse complète Phase 5: Tout Phase 4 + Réponse naturelle générée"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "user_id": 123,
                "message": "Combien j'ai dépensé chez Amazon ce mois ?",
                "timestamp": "2024-08-26T14:30:00Z",
                "processing_time_ms": 1678,
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
                "response": {
                    "message": "Ce mois-ci, vous avez dépensé **234,56€** chez Amazon sur **12 transactions**.",
                    "structured_data": {
                        "total_amount": 234.56,
                        "currency": "EUR",
                        "transaction_count": 12
                    },
                    "insights": [
                        {
                            "type": "trend",
                            "title": "Augmentation mensuelle",
                            "description": "Vos dépenses Amazon ont augmenté de 24% ce mois",
                            "severity": "info"
                        }
                    ]
                },
                "phase": 5
            }
        }
    )
    
    # Phase 5 - Réponse générée
    response: Optional[ResponseContent] = None
    response_quality: Optional[ResponseQuality] = None
    response_generation_metrics: Optional[ResponseGenerationMetrics] = None
    
    # Override phase par défaut
    phase: int = 5
    
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
    
    @computed_field
    @property
    def performance_summary_phase5(self) -> Dict[str, Any]:
        """Résumé performance Phase 5 complet"""
        base_summary = self.performance_summary_phase4
        
        phase5_summary = {
            **base_summary,
            "phase": 5,
            "response_generation_success": self.response_generation_success,
            "has_response": self.has_response,
            "total_time_with_response_ms": self.total_processing_time_with_response_ms
        }
        
        # Ajouter métriques génération réponse
        if self.response_generation_metrics:
            phase5_summary.update({
                "response_generation_time_ms": self.response_generation_metrics.generation_time_ms,
                "response_quality_score": self.response_generation_metrics.quality_score,
                "insights_generated": self.response_generation_metrics.insights_generated,
                "suggestions_generated": self.response_generation_metrics.suggestions_generated,
                "personalization_applied": self.response_generation_metrics.personalization_applied
            })
        
        # Ajouter métriques qualité
        if self.response_quality:
            phase5_summary.update({
                "response_relevance": self.response_quality.relevance_score,
                "response_completeness": self.response_quality.completeness,
                "response_actionability": self.response_quality.actionability,
                "response_tone": self.response_quality.tone
            })
        
        return phase5_summary
    
    def set_response_content(self, content: ResponseContent) -> None:
        """Définit le contenu de la réponse"""
        self.response = content
    
    def set_response_quality(self, quality: ResponseQuality) -> None:
        """Définit la qualité de la réponse"""
        self.response_quality = quality
    
    def set_response_generation_metrics(self, metrics: ResponseGenerationMetrics) -> None:
        """Définit les métriques de génération"""
        self.response_generation_metrics = metrics
    
    def to_phase4_response(self) -> ConversationResponsePhase4:
        """Conversion vers réponse Phase 4 (sans réponse générée)"""
        return ConversationResponsePhase4(
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
            agent_metrics_detailed=self.agent_metrics_detailed,
            search_results=self.search_results,
            resilience_metrics=self.resilience_metrics,
            search_metrics=self.search_metrics
        )
    
    def to_minimal_dict_phase5(self) -> Dict[str, Any]:
        """Version minimale Phase 5 pour logs"""
        base_minimal = self.to_minimal_dict_phase4()
        return {
            **base_minimal,
            "phase": 5,
            "has_response": self.has_response,
            "response_generation_success": self.response_generation_success,
            "insights_count": len(self.response.insights) if self.response else 0,
            "suggestions_count": len(self.response.suggestions) if self.response else 0,
            "response_quality_score": self.response_quality.relevance_score if self.response_quality else 0.0
        }


class ConversationResponseFactoryPhase5:
    """Factory pour créer les réponses Phase 5"""
    
    @staticmethod
    def create_phase5_success(
        base_response: ConversationResponsePhase4,
        response_content: ResponseContent,
        response_quality: ResponseQuality,
        generation_metrics: ResponseGenerationMetrics
    ) -> ConversationResponsePhase5:
        """Création réponse Phase 5 réussie avec réponse générée"""
        
        response = ConversationResponsePhase5(
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
            response_generation_metrics=generation_metrics
        )
        
        return response
    
    @staticmethod
    def create_phase5_from_phase4(
        phase4_response: ConversationResponsePhase4
    ) -> ConversationResponsePhase5:
        """Création Phase 5 depuis Phase 4 (sans réponse générée encore)"""
        
        return ConversationResponsePhase5(
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
            search_metrics=phase4_response.search_metrics
        )