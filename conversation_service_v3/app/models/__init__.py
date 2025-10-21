"""
Models for conversation_service_v3
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AgentRole(Enum):
    """Rôles des agents dans le pipeline"""
    INTENT_ROUTER = "intent_router"
    QUERY_ANALYZER = "query_analyzer"
    ELASTICSEARCH_BUILDER = "elasticsearch_builder"
    QUERY_VALIDATOR = "query_validator"
    RESPONSE_GENERATOR = "response_generator"


@dataclass
class UserQuery:
    """Requête utilisateur brute"""
    user_id: int
    message: str
    conversation_id: Optional[str] = None
    context: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Résultat de l'analyse de la requête utilisateur"""
    intent: str  # "search", "aggregate", "compare", "analyze"
    entities: Dict[str, Any]  # Entités extraites
    filters: Dict[str, Any]  # Filtres à appliquer
    aggregations_needed: List[str]  # Types d'agrégations nécessaires
    time_range: Optional[Dict[str, str]] = None
    confidence: float = 0.0


@dataclass
class ElasticsearchQuery:
    """Query Elasticsearch construite"""
    query: Dict[str, Any]
    aggregations: Optional[Dict[str, Any]] = None
    size: int = 50  # Nombre de documents à récupérer
    sort: Optional[List[Dict[str, str]]] = None


@dataclass
class QueryValidationResult:
    """Résultat de validation d'une query"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_fix: Optional[Dict[str, Any]] = None


@dataclass
class SearchResults:
    """Résultats de recherche Elasticsearch"""
    hits: List[Dict[str, Any]]
    total: int
    aggregations: Optional[Dict[str, Any]] = None
    took_ms: int = 0


@dataclass
class AgentResponse:
    """Réponse d'un agent"""
    success: bool
    data: Any
    agent_role: AgentRole
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationResponse:
    """Réponse finale au utilisateur"""
    success: bool
    message: str
    search_results: Optional[SearchResults] = None
    aggregations_summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
