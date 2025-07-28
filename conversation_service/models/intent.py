"""
📋 Modèles Pydantic - Structures de données API

Modèles de requêtes et réponses pour l'API de détection d'intention.
Basés sur le code fonctionnel existant avec extensions modulaires.
Version Pydantic V2 avec field_validator et model_validator.
"""

import time
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from .enums import IntentType, EntityType, DetectionMethod, ConfidenceLevel


class IntentRequest(BaseModel):
    """Requête de détection d'intention"""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="Texte utilisateur à analyser"
    )
    user_id: Optional[Union[str, int]] = Field(
        None, 
        description="Identifiant utilisateur optionnel"
    )
    use_deepseek_fallback: bool = Field(
        True, 
        description="Autoriser fallback DeepSeek si règles insuffisantes"
    )
    force_method: Optional[DetectionMethod] = Field(
        None,
        description="Forcer utilisation méthode spécifique (debug)"
    )
    enable_cache: bool = Field(
        True,
        description="Utiliser cache si disponible"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Contexte conversationnel optionnel"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validation et nettoyage basique de la requête"""
        if not v or not v.strip():
            raise ValueError("Query ne peut pas être vide")
        return v.strip()
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
        """Validation user_id"""
        if v is not None:
            if isinstance(v, str) and not v.strip():
                raise ValueError("user_id string ne peut pas être vide")
            if isinstance(v, int) and v <= 0:
                raise ValueError("user_id integer doit être positif")
        return v


class IntentResponse(BaseModel):
    """Réponse de détection d'intention"""
    
    intent: IntentType = Field(..., description="Intention détectée")
    intent_code: str = Field(..., description="Code intention pour search service")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Score de confiance [0-1]"
    )
    processing_time_ms: float = Field(..., description="Temps traitement en ms")
    method_used: DetectionMethod = Field(..., description="Méthode de détection utilisée")
    query: str = Field(..., description="Requête originale")
    entities: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Entités extraites"
    )
    suggestions: List[str] = Field(
        default_factory=list, 
        description="Suggestions contextuelles"
    )
    cost_estimate: float = Field(
        default=0.0, 
        description="Coût estimé de la requête"
    )
    confidence_level: Optional[ConfidenceLevel] = Field(
        None,
        description="Niveau de confiance catégorisé"
    )
    cached: bool = Field(default=False, description="Résultat provient du cache")
    
    @model_validator(mode='after')
    def set_confidence_level(self) -> 'IntentResponse':
        """Calcule automatiquement le niveau de confiance"""
        if self.confidence_level is None:
            self.confidence_level = ConfidenceLevel.from_score(self.confidence)
        return self


class EntityExtractionRequest(BaseModel):
    """Requête d'extraction d'entités seule"""
    
    query: str = Field(..., min_length=1, max_length=500)
    intent_context: Optional[IntentType] = Field(
        None,
        description="Contexte d'intention pour extraction ciblée"
    )
    expected_entities: Optional[List[EntityType]] = Field(
        None,
        description="Types d'entités attendues"
    )


class EntityExtractionResponse(BaseModel):
    """Réponse d'extraction d'entités"""
    
    entities: Dict[str, Any] = Field(..., description="Entités extraites")
    confidence_per_entity: Dict[str, float] = Field(
        default_factory=dict,
        description="Confiance par entité extraite"
    )
    processing_time_ms: float = Field(..., description="Temps traitement")
    method_used: str = Field(default="pattern_matching", description="Méthode extraction")
    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées détaillées d'extraction"
    )


class BatchIntentRequest(BaseModel):
    """Requête de traitement par batch"""
    
    queries: List[str] = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="Liste des requêtes à traiter"
    )
    user_id: Optional[Union[str, int]] = None
    use_deepseek_fallback: bool = True
    enable_cache: bool = True
    parallel_processing: bool = Field(
        True,
        description="Traitement parallèle des requêtes"
    )
    
    @field_validator('queries')
    @classmethod
    def validate_queries(cls, v: List[str]) -> List[str]:
        """Validation des requêtes batch"""
        if not v:
            raise ValueError("Liste de requêtes ne peut pas être vide")
        
        # Validation de chaque requête
        validated_queries = []
        for i, query in enumerate(v):
            if not query or not query.strip():
                raise ValueError(f"Requête {i} ne peut pas être vide")
            if len(query) > 500:
                raise ValueError(f"Requête {i} trop longue (max 500 caractères)")
            validated_queries.append(query.strip())
        
        return validated_queries


class BatchIntentResponse(BaseModel):
    """Réponse de traitement par batch"""
    
    results: List[IntentResponse] = Field(..., description="Résultats par requête")
    batch_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métriques du batch"
    )
    total_processing_time_ms: float = Field(..., description="Temps total traitement")
    successful_requests: int = Field(..., description="Nombre requêtes réussies")
    failed_requests: int = Field(default=0, description="Nombre requêtes échouées")


class HealthResponse(BaseModel):
    """Réponse de santé du service"""
    
    status: str = Field(default="healthy", description="Statut général")
    service_name: str = Field(default="conversation-service")
    version: str = Field(default="2.0.0")
    timestamp: float = Field(default_factory=time.time)
    
    # Statuts des composants
    rule_engine_status: str = Field(default="operational")
    deepseek_client_status: str = Field(default="unknown")
    cache_status: str = Field(default="operational")
    
    # Métriques de base
    total_requests: int = Field(default=0)
    average_latency_ms: float = Field(default=0.0)
    cache_hit_rate: float = Field(default=0.0)
    
    # Configuration active
    deepseek_fallback_enabled: bool = Field(default=True)
    cache_enabled: bool = Field(default=True)


class MetricsResponse(BaseModel):
    """Réponse détaillée des métriques"""
    
    # Métriques de performance
    total_requests: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)
    
    # Métriques de qualité
    performance: Dict[str, Any] = Field(default_factory=dict)
    distribution: Dict[str, int] = Field(default_factory=dict)
    cache_metrics: Dict[str, Any] = Field(default_factory=dict)
    efficiency: Dict[str, float] = Field(default_factory=dict)
    
    # Métriques temporelles
    last_hour_requests: int = Field(default=0)
    last_hour_avg_latency: float = Field(default=0.0)
    
    # Statuts des seuils
    meets_latency_target: bool = Field(default=True)
    meets_accuracy_target: bool = Field(default=True)
    target_latency_ms: float = Field(default=50.0)
    target_accuracy: float = Field(default=0.85)


class IntentSuggestion(BaseModel):
    """Suggestion d'intention avec contexte"""
    
    suggestion_text: str = Field(..., description="Texte de la suggestion")
    target_intent: IntentType = Field(..., description="Intention cible")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance suggestion")
    context_relevant: bool = Field(default=True, description="Pertinence contextuelle")


class IntentClassificationResult(BaseModel):
    """Résultat détaillé de classification (pour debug)"""
    
    intent: IntentType
    confidence: float
    method: DetectionMethod
    rule_scores: Optional[Dict[str, float]] = None
    ml_scores: Optional[Dict[str, float]] = None
    llm_response: Optional[str] = None
    entities_raw: Optional[Dict[str, Any]] = None
    processing_steps: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Réponse d'erreur structurée"""
    
    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur détaillé") 
    details: Optional[Dict[str, Any]] = Field(None, description="Détails techniques")
    timestamp: float = Field(default_factory=time.time)
    request_id: Optional[str] = Field(None, description="ID de la requête échouée")


# Modèles utilitaires pour cache et métriques
class CacheEntry(BaseModel):
    """Entrée du cache avec métadonnées"""
    
    query: str
    result: IntentResponse
    created_at: float = Field(default_factory=time.time)
    hit_count: int = Field(default=1)
    last_accessed: float = Field(default_factory=time.time)


class RequestMetadata(BaseModel):
    """Métadonnées d'une requête pour analytics"""
    
    request_id: str = Field(..., description="ID unique requête")
    user_id: Optional[Union[str, int]] = None
    timestamp: float = Field(default_factory=time.time)
    query_length: int = Field(..., description="Longueur caractères")
    detected_intent: IntentType
    confidence: float
    method_used: DetectionMethod
    processing_time_ms: float
    cost: float = Field(default=0.0)
    cached: bool = Field(default=False)
    entities_count: int = Field(default=0)


# Modèles avancés pour API enrichie
class DetailedIntentResponse(IntentResponse):
    """Réponse enrichie avec détails de traitement"""
    
    processing_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Détails du processus de détection"
    )
    rule_matches: List[str] = Field(
        default_factory=list,
        description="Règles qui ont matché"
    )
    fallback_used: bool = Field(
        default=False,
        description="Fallback LLM utilisé"
    )
    entity_extraction_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Détails extraction d'entités"
    )


class ConversationContext(BaseModel):
    """Contexte conversationnel pour requêtes multi-tours"""
    
    conversation_id: str = Field(..., description="ID unique conversation")
    previous_intents: List[IntentType] = Field(
        default_factory=list,
        description="Intentions précédentes dans la conversation"
    )
    user_preferences: Optional[Dict[str, Any]] = Field(
        None,
        description="Préférences utilisateur"
    )
    session_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entités persistantes dans la session"
    )
    turn_number: int = Field(default=1, description="Numéro du tour de parole")


class IntentWithContext(IntentRequest):
    """Requête d'intention avec contexte conversationnel"""
    
    conversation_context: Optional[ConversationContext] = Field(
        None,
        description="Contexte conversationnel"
    )
    
    @model_validator(mode='after')
    def validate_context_consistency(self) -> 'IntentWithContext':
        """Valide la cohérence du contexte conversationnel"""
        if self.conversation_context and self.context:
            # Merge des contextes si les deux sont présents
            if 'conversation_id' not in self.context:
                self.context['conversation_id'] = self.conversation_context.conversation_id
        return self


class BulkProcessingRequest(BaseModel):
    """Requête de traitement en masse avec options avancées"""
    
    requests: List[Union[IntentRequest, IntentWithContext]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Requêtes à traiter en masse"
    )
    processing_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Options de traitement"
    )
    priority: int = Field(
        default=0,
        ge=-10,
        le=10,
        description="Priorité de traitement (-10 à 10)"
    )
    
    @field_validator('requests')
    @classmethod
    def validate_bulk_requests(cls, v: List[Union[IntentRequest, IntentWithContext]]) -> List[Union[IntentRequest, IntentWithContext]]:
        """Validation des requêtes bulk"""
        if not v:
            raise ValueError("Liste de requêtes ne peut pas être vide")
        
        # Vérification de diversité pour éviter spam
        unique_queries = set()
        for req in v:
            if req.query in unique_queries:
                raise ValueError(f"Requête dupliquée détectée: {req.query}")
            unique_queries.add(req.query)
        
        return v


class ServiceHealthCheck(BaseModel):
    """Check de santé avancé du service"""
    
    overall_status: str = Field(..., description="Statut global")
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Statut détaillé des composants"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Métriques de performance temps réel"
    )
    last_check_timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp du dernier check"
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Alertes actives"
    )
    
    @model_validator(mode='after')
    def determine_overall_status(self) -> 'ServiceHealthCheck':
        """Détermine le statut global basé sur les composants"""
        if not self.components:
            self.overall_status = "unknown"
            return self
        
        critical_failures = []
        warnings = []
        
        for component, status_info in self.components.items():
            status = status_info.get("status", "unknown")
            if status in ["error", "failed", "critical"]:
                critical_failures.append(component)
            elif status in ["warning", "degraded"]:
                warnings.append(component)
        
        if critical_failures:
            self.overall_status = "unhealthy"
            self.alerts.extend([f"Component {comp} is critical" for comp in critical_failures])
        elif warnings:
            self.overall_status = "degraded"
            self.alerts.extend([f"Component {comp} has warnings" for comp in warnings])
        else:
            self.overall_status = "healthy"
        
        return self


# Exports publics
__all__ = [
    # Modèles de requête principaux
    "IntentRequest",
    "EntityExtractionRequest", 
    "BatchIntentRequest",
    "IntentWithContext",
    "BulkProcessingRequest",
    
    # Modèles de réponse principaux
    "IntentResponse",
    "EntityExtractionResponse",
    "BatchIntentResponse",
    "DetailedIntentResponse",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse",
    "ServiceHealthCheck",
    
    # Modèles utilitaires
    "IntentSuggestion",
    "IntentClassificationResult",
    "CacheEntry",
    "RequestMetadata",
    "ConversationContext"
]