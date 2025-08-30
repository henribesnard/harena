"""
Modèles Pydantic Phase 2 : Réponses Conversation Étendues
Extension des réponses conversation avec extraction entités avancée
100% compatible Phase 1 + enrichissements AutoGen multi-agents
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# Imports Phase 1 (compatibilité)
from conversation_service.models.responses.conversation_responses import (
    ConversationResponse, IntentClassificationResult
)
from conversation_service.models.responses.enriched_conversation_responses import (
    EnrichedConversationResponse, AutoGenTeamMetadata, ProcessingMode
)

# Imports Phase 2 nouveaux modèles
from conversation_service.models.conversation.entities import (
    ComprehensiveEntityExtraction, EntityExtractionConfidence,
    ExtractedAmount, ExtractedMerchant, ExtractedDateRange, ExtractedCategory
)


class EntityValidationResult(BaseModel):
    """Résultat validation croisée entités par équipe AutoGen"""
    
    validation_id: str = Field(default_factory=lambda: str(uuid4()))
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Scores validation
    coherence_score: float = Field(..., description="Score cohérence entités", ge=0.0, le=1.0)
    completeness_score: float = Field(..., description="Score complétude extraction", ge=0.0, le=1.0)
    business_logic_score: float = Field(..., description="Score logique métier", ge=0.0, le=1.0)
    
    # Flags validation
    entities_coherent: bool = Field(default=False, description="Entités cohérentes entre elles")
    extraction_complete: bool = Field(default=False, description="Extraction complète détectée")
    business_rules_respected: bool = Field(default=True, description="Règles métier respectées")
    
    # Détails validation
    validation_issues: List[str] = Field(default_factory=list, description="Issues détectées")
    validation_warnings: List[str] = Field(default_factory=list, description="Warnings non bloquants")
    cross_validation_details: Dict[str, Any] = Field(default_factory=dict, description="Détails validation croisée")
    
    # Recommandations
    recommended_actions: List[str] = Field(default_factory=list, description="Actions recommandées")
    confidence_adjustment: float = Field(default=0.0, description="Ajustement confiance suggéré", ge=-0.5, le=0.5)
    
    @field_validator('coherence_score', 'completeness_score', 'business_logic_score')
    def validate_scores(cls, v):
        """Validation scores 0-1"""
        return max(0.0, min(1.0, v))
    
    @model_validator(mode='after')
    def calculate_validation_flags(self):
        """Calcul flags depuis scores"""
        coherence = self.coherence_score
        completeness = self.completeness_score
        business = self.business_logic_score
        
        self.entities_coherent = coherence >= 0.7
        self.extraction_complete = completeness >= 0.8
        self.business_rules_respected = business >= 0.9
        
        return self
    
    def get_overall_validation_score(self) -> float:
        """Score validation global pondéré"""
        return (
            self.coherence_score * 0.4 +
            self.completeness_score * 0.3 +
            self.business_logic_score * 0.3
        )
    
    def is_validation_passed(self, min_score: float = 0.75) -> bool:
        """Validation passée selon seuil"""
        return self.get_overall_validation_score() >= min_score


class MultiAgentProcessingInsights(BaseModel):
    """Insights détaillés traitement multi-agents AutoGen"""
    
    # Métadonnées traitement
    processing_session_id: str = Field(default_factory=lambda: str(uuid4()))
    agents_coordination: Dict[str, Any] = Field(default_factory=dict, description="Détails coordination agents")
    workflow_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Étapes workflow")
    
    # Performance agents individuels
    intent_agent_performance: Dict[str, Any] = Field(default_factory=dict, description="Performance agent intent")
    entity_agent_performance: Dict[str, Any] = Field(default_factory=dict, description="Performance agent entités")
    orchestrator_performance: Dict[str, Any] = Field(default_factory=dict, description="Performance orchestrateur")
    
    # Interactions agents
    agent_interactions: List[Dict[str, Any]] = Field(default_factory=list, description="Historique interactions")
    consensus_reached: bool = Field(default=False, description="Consensus agents atteint")
    conflicting_results: List[Dict[str, Any]] = Field(default_factory=list, description="Résultats conflictuels")
    
    # Métriques qualité
    iteration_count: int = Field(default=1, description="Nombre itérations", ge=1)
    convergence_achieved: bool = Field(default=False, description="Convergence atteinte")
    quality_improvement_ratio: float = Field(default=0.0, description="Ratio amélioration qualité")
    
    # Temps traitement détaillé
    intent_processing_time_ms: int = Field(default=0, description="Temps agent intent ms")
    entity_processing_time_ms: int = Field(default=0, description="Temps agent entités ms")
    coordination_overhead_ms: int = Field(default=0, description="Overhead coordination ms")
    
    def get_total_agent_time(self) -> int:
        """Temps total agents (sans overhead)"""
        return self.intent_processing_time_ms + self.entity_processing_time_ms
    
    def get_efficiency_ratio(self) -> float:
        """Ratio efficacité (temps agents / temps total)"""
        total_time = self.get_total_agent_time() + self.coordination_overhead_ms
        if total_time == 0:
            return 1.0
        return self.get_total_agent_time() / total_time


class ConversationResponsePhase2(EnrichedConversationResponse):
    """
    Réponse conversation Phase 2 - Extension complète avec entités avancées
    Hérite EnrichedConversationResponse (Phase 1.5) pour compatibilité totale
    """
    
    # Extensions Phase 2
    comprehensive_entities: ComprehensiveEntityExtraction = Field(
        ..., description="Extraction entités complète multi-agents"
    )
    
    entity_validation: EntityValidationResult = Field(
        ..., description="Résultat validation croisée entités"
    )
    
    multi_agent_insights: MultiAgentProcessingInsights = Field(
        ..., description="Insights détaillés traitement multi-agents"
    )
    
    # Métadonnées Phase 2
    phase_version: Literal["phase2"] = Field(default="phase2", description="Version phase modèle")
    advanced_features_enabled: bool = Field(default=True, description="Fonctions avancées actives")
    
    # Compatibilité/fallback
    fallback_reason: Optional[str] = Field(None, description="Raison fallback si applicable")
    single_agent_backup: Optional[EnrichedConversationResponse] = Field(
        None, description="Réponse agent unique backup"
    )
    
    @model_validator(mode='after')
    def ensure_compatibility(self):
        """Assurer compatibilité formats précédents"""
        
        # Synchroniser entities_extraction avec comprehensive_entities
        comprehensive = self.comprehensive_entities
        if comprehensive:
            # Mise à jour format Phase 1.5 pour compatibilité
            legacy_entities = comprehensive.to_legacy_entities_dict()
            
            # Si entities_extraction pas défini, créer depuis comprehensive
            if not self.entities_extraction:
                # Créer extraction basique depuis comprehensive pour compatibilité
                from conversation_service.models.responses.enriched_conversation_responses import EntityExtractionResult
                self.entities_extraction = EntityExtractionResult(
                    entities=legacy_entities,
                    confidence=comprehensive.overall_confidence,
                    entities_count=legacy_entities.get('entities_count', 0)
                )
        
        return self
    
    @field_validator('comprehensive_entities')
    def validate_comprehensive_entities(cls, v):
        """Validation entités complètes"""
        if not v.entities_found:
            # Log warning mais ne pas échouer
            pass
        return v
    
    def get_best_confidence_entities(self) -> Dict[str, List[Any]]:
        """Entités avec meilleure confiance pour réponse utilisateur"""
        return self.comprehensive_entities.get_high_confidence_entities(min_confidence=0.8)
    
    def is_high_quality_response(self) -> bool:
        """Réponse haute qualité selon critères Phase 2"""
        return (
            self.comprehensive_entities.overall_confidence >= 0.8 and
            self.entity_validation.get_overall_validation_score() >= 0.75 and
            self.multi_agent_insights.consensus_reached and
            len(self.entity_validation.validation_issues) == 0
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Résumé traitement pour monitoring"""
        return {
            "processing_mode": self.autogen_metadata.processing_mode,
            "entities_found": self.comprehensive_entities.entities_found,
            "high_confidence_entities": self.comprehensive_entities.high_confidence_entities,
            "validation_passed": self.entity_validation.is_validation_passed(),
            "consensus_reached": self.multi_agent_insights.consensus_reached,
            "total_processing_time_ms": self.processing_time_ms,
            "quality_score": self.entity_validation.get_overall_validation_score(),
            "is_high_quality": self.is_high_quality_response()
        }
    
    # Factory Methods Phase 2
    
    @classmethod
    def from_multi_agent_results(
        cls,
        user_id: int,
        message: str,
        intent: IntentClassificationResultResult,
        comprehensive_entities: ComprehensiveEntityExtraction,
        entity_validation: EntityValidationResult,
        multi_agent_insights: MultiAgentProcessingInsights,
        processing_time_ms: int,
        **kwargs
    ) -> 'ConversationResponsePhase2':
        """Factory depuis résultats multi-agents complets"""
        
        # Créer métadonnées AutoGen
        autogen_metadata = AutoGenTeamMetadata(
            processing_mode=ProcessingMode.MULTI_AGENT_TEAM,
            workflow_status="completed",
            agents_involved=["intent_classifier", "entity_extractor"],
            coherence_validation={
                "score": entity_validation.coherence_score,
                "threshold_met": entity_validation.entities_coherent
            }
        )
        
        return cls(
            user_id=user_id,
            message=message,
            intent=intent,
            processing_time_ms=processing_time_ms,
            status="success",
            comprehensive_entities=comprehensive_entities,
            entity_validation=entity_validation,
            multi_agent_insights=multi_agent_insights,
            autogen_metadata=autogen_metadata,
            **kwargs
        )
    
    @classmethod
    def from_single_agent_fallback(
        cls,
        enriched_response: EnrichedConversationResponse,
        fallback_reason: str
    ) -> 'ConversationResponsePhase2':
        """Factory depuis fallback agent unique"""
        
        # Créer entités minimales depuis réponse enrichie
        minimal_entities = ComprehensiveEntityExtraction(
            user_message=enriched_response.message,
            overall_confidence=0.5,  # Confiance réduite fallback
            extraction_method="single_agent"
        )
        
        # Validation basique
        basic_validation = EntityValidationResult(
            coherence_score=0.6,
            completeness_score=0.4,
            business_logic_score=0.8,
            validation_issues=[f"Fallback reason: {fallback_reason}"]
        )
        
        # Insights basiques
        basic_insights = MultiAgentProcessingInsights(
            consensus_reached=False,
            iteration_count=1,
            intent_processing_time_ms=enriched_response.processing_time_ms
        )
        
        return cls(
            user_id=enriched_response.user_id,
            message=enriched_response.message,
            intent=enriched_response.intent,
            processing_time_ms=enriched_response.processing_time_ms,
            status=enriched_response.status,
            comprehensive_entities=minimal_entities,
            entity_validation=basic_validation,
            multi_agent_insights=basic_insights,
            autogen_metadata=enriched_response.autogen_metadata,
            fallback_reason=fallback_reason,
            single_agent_backup=enriched_response,
            advanced_features_enabled=False
        )


class BatchProcessingResponse(BaseModel):
    """Réponse traitement batch messages multiples"""
    
    batch_id: str = Field(default_factory=lambda: str(uuid4()))
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Résultats batch
    responses: List[ConversationResponsePhase2] = Field(..., description="Réponses individuelles")
    batch_size: int = Field(..., description="Taille batch")
    successful_responses: int = Field(default=0, description="Réponses réussies")
    failed_responses: int = Field(default=0, description="Réponses échouées")
    
    # Métriques batch
    total_processing_time_ms: int = Field(default=0, description="Temps traitement total")
    average_response_time_ms: float = Field(default=0.0, description="Temps moyen par réponse")
    batch_efficiency_score: float = Field(default=0.0, description="Score efficacité batch")
    
    # Insights batch
    common_entities: Dict[str, int] = Field(default_factory=dict, description="Entités communes détectées")
    processing_patterns: Dict[str, Any] = Field(default_factory=dict, description="Patterns traitement")
    
    @model_validator(mode='after')
    def calculate_batch_metrics(self):
        """Calcul métriques batch"""
        responses = self.responses
        
        self.batch_size = len(responses)
        self.successful_responses = sum(1 for r in responses if r.status == "success")
        self.failed_responses = self.batch_size - self.successful_responses
        
        if responses:
            total_time = sum(r.processing_time_ms for r in responses)
            self.total_processing_time_ms = total_time
            self.average_response_time_ms = total_time / len(responses)
            
            # Score efficacité basé sur succès et performance
            success_ratio = self.successful_responses / self.batch_size
            avg_quality = sum(r.entity_validation.get_overall_validation_score() 
                            for r in responses if r.status == "success")
            avg_quality = avg_quality / self.successful_responses if self.successful_responses > 0 else 0
            
            self.batch_efficiency_score = (success_ratio * 0.6) + (avg_quality * 0.4)
        
        return self


# Factory Functions Phase 2

def create_phase2_response_from_entities(
    user_id: int,
    message: str,
    intent: IntentClassificationResultResult,
    amounts: List[ExtractedAmount] = None,
    merchants: List[ExtractedMerchant] = None,
    date_ranges: List[ExtractedDateRange] = None,
    categories: List[ExtractedCategory] = None,
    processing_time_ms: int = 1000
) -> ConversationResponsePhase2:
    """Factory création réponse Phase 2 depuis entités individuelles"""
    
    # Créer extraction comprehensive
    comprehensive_entities = ComprehensiveEntityExtraction(
        user_message=message,
        amounts=amounts or [],
        merchants=merchants or [],
        date_ranges=date_ranges or [],
        categories=categories or []
    )
    
    # Validation basique
    validation = EntityValidationResult(
        coherence_score=0.8,
        completeness_score=0.7,
        business_logic_score=0.9
    )
    
    # Insights basiques
    insights = MultiAgentProcessingInsights(
        consensus_reached=True,
        iteration_count=1,
        intent_processing_time_ms=processing_time_ms // 2,
        entity_processing_time_ms=processing_time_ms // 2
    )
    
    return ConversationResponsePhase2.from_multi_agent_results(
        user_id=user_id,
        message=message,
        intent=intent,
        comprehensive_entities=comprehensive_entities,
        entity_validation=validation,
        multi_agent_insights=insights,
        processing_time_ms=processing_time_ms
    )


def create_minimal_phase2_response(
    user_id: int,
    message: str,
    intent: IntentClassificationResult
) -> ConversationResponsePhase2:
    """Factory réponse Phase 2 minimale (pour tests)"""
    
    return create_phase2_response_from_entities(
        user_id=user_id,
        message=message,
        intent=intent
    )