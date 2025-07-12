"""
Modèles de requêtes internes du Search Service
Structures simplifiées pour le traitement interne des contrats
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass

from .service_contracts import (
    QueryType, 
    FilterOperator, 
    AggregationType,
    SearchServiceQuery
)


class InternalQueryType(str, Enum):
    """Types de requêtes internes simplifiés"""
    SIMPLE_TERM = "simple_term"           # Filtre simple sur un champ
    MULTI_TERM = "multi_term"             # Filtres multiples AND
    TEXT_MATCH = "text_match"             # Recherche textuelle pure
    HYBRID_SEARCH = "hybrid_search"       # Texte + filtres
    AGGREGATION_ONLY = "aggregation_only" # Seulement agrégations
    COMPLEX_QUERY = "complex_query"       # Requête complexe multi-critères


class ProcessingMode(str, Enum):
    """Modes de traitement des requêtes"""
    FAST = "fast"           # Optimisé vitesse, cache agressif
    BALANCED = "balanced"   # Équilibre performance/précision
    PRECISE = "precise"     # Optimisé précision, cache minimal


class CacheStrategy(str, Enum):
    """Stratégies de cache"""
    AGGRESSIVE = "aggressive"  # Cache tout longtemps
    STANDARD = "standard"      # Cache standard 5min
    MINIMAL = "minimal"        # Cache court 1min
    DISABLED = "disabled"      # Pas de cache


# === MODÈLES INTERNES SIMPLIFIÉS ===

@dataclass
class FieldBoost:
    """Configuration boost pour un champ"""
    field: str
    boost: float = 1.0
    
    def __post_init__(self):
        if self.boost < 0.1 or self.boost > 10.0:
            raise ValueError("boost doit être entre 0.1 et 10.0")


@dataclass 
class TermFilter:
    """Filtre terme interne simplifié"""
    field: str
    value: Union[str, int, float, List[Any]]
    operator: FilterOperator = FilterOperator.EQ
    boost: float = 1.0
    
    def to_elasticsearch_term(self) -> Dict[str, Any]:
        """Convertit en terme Elasticsearch"""
        if self.operator == FilterOperator.EQ:
            return {"term": {self.field: {"value": self.value, "boost": self.boost}}}
        elif self.operator == FilterOperator.IN:
            return {"terms": {self.field: self.value, "boost": self.boost}}
        elif self.operator in [FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE]:
            return {"range": {self.field: {self.operator.value: self.value}}}
        elif self.operator == FilterOperator.BETWEEN:
            return {"range": {self.field: {"gte": self.value[0], "lte": self.value[1]}}}
        else:
            raise ValueError(f"Opérateur non supporté: {self.operator}")


@dataclass
class TextQuery:
    """Requête textuelle interne"""
    query: str
    fields: List[str]
    field_boosts: List[FieldBoost]
    fuzziness: Optional[str] = None
    minimum_should_match: Optional[str] = None
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convertit en requête Elasticsearch"""
        # Construire les champs avec boost
        boosted_fields = []
        for field in self.fields:
            # Chercher le boost pour ce champ
            boost = 1.0
            for field_boost in self.field_boosts:
                if field_boost.field == field:
                    boost = field_boost.boost
                    break
            
            if boost != 1.0:
                boosted_fields.append(f"{field}^{boost}")
            else:
                boosted_fields.append(field)
        
        query_config = {
            "query": self.query,
            "fields": boosted_fields
        }
        
        if self.fuzziness:
            query_config["fuzziness"] = self.fuzziness
        if self.minimum_should_match:
            query_config["minimum_should_match"] = self.minimum_should_match
            
        return {"multi_match": query_config}


class InternalSearchRequest(BaseModel):
    """Requête de recherche interne simplifiée"""
    
    # Identification
    request_id: str = Field(..., description="ID unique de la requête")
    user_id: int = Field(..., description="ID utilisateur (sécurité)")
    
    # Type et mode
    query_type: InternalQueryType = Field(..., description="Type de requête interne")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.BALANCED, description="Mode de traitement")
    
    # Filtres simplifiés
    term_filters: List[TermFilter] = Field(default_factory=list, description="Filtres terme")
    text_query: Optional[TextQuery] = Field(default=None, description="Requête textuelle")
    
    # Paramètres de recherche
    limit: int = Field(default=20, description="Nombre de résultats")
    offset: int = Field(default=0, description="Décalage pagination")
    min_score: Optional[float] = Field(default=None, description="Score minimum")
    
    # Agrégations simplifiées
    aggregation_fields: List[str] = Field(default_factory=list, description="Champs à agréger")
    aggregation_types: List[AggregationType] = Field(default_factory=list, description="Types d'agrégation")
    
    # Options de traitement
    include_highlights: bool = Field(default=False, description="Inclure highlights")
    cache_strategy: CacheStrategy = Field(default=CacheStrategy.STANDARD, description="Stratégie de cache")
    timeout_ms: int = Field(default=5000, description="Timeout")
    
    # Métadonnées de traçabilité (optionnelles pour interne)
    original_query_id: Optional[str] = Field(default=None, description="ID requête originale")
    agent_context: Optional[Dict[str, str]] = Field(default=None, description="Contexte agent")
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        """Valide l'ID utilisateur"""
        if v <= 0:
            raise ValueError("user_id doit être positif")
        return v
    
    @field_validator("term_filters")
    @classmethod
    def ensure_user_filter_exists(cls, v, info):
        """S'assure qu'un filtre user_id existe"""
        if not info.data:
            return v
            
        user_id = info.data.get("user_id")
        if not user_id:
            return v
            
        # Vérifier si user_id filter existe déjà
        has_user_filter = any(f.field == "user_id" for f in v)
        
        if not has_user_filter:
            # Ajouter automatiquement le filtre user_id pour sécurité
            user_filter = TermFilter(
                field="user_id",
                value=user_id,
                operator=FilterOperator.EQ
            )
            v.append(user_filter)
        
        return v
    
    def get_cache_key(self) -> str:
        """Génère une clé de cache unique pour cette requête"""
        import hashlib
        import json
        
        # Éléments qui définissent l'unicité de la requête
        cache_elements = {
            "user_id": self.user_id,
            "query_type": self.query_type.value,
            "term_filters": [
                {
                    "field": f.field,
                    "value": f.value, 
                    "operator": f.operator.value
                } for f in self.term_filters
            ],
            "text_query": {
                "query": self.text_query.query,
                "fields": self.text_query.fields
            } if self.text_query else None,
            "limit": self.limit,
            "offset": self.offset,
            "aggregation_fields": sorted(self.aggregation_fields),
            "aggregation_types": sorted([t.value for t in self.aggregation_types])
        }
        
        # Créer hash stable
        cache_str = json.dumps(cache_elements, sort_keys=True)
        return f"search:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def get_estimated_complexity(self) -> str:
        """Estime la complexité de la requête"""
        complexity_score = 0
        
        # Score basé sur les filtres
        complexity_score += len(self.term_filters)
        
        # Score basé sur la recherche textuelle
        if self.text_query:
            complexity_score += 2
            complexity_score += len(self.text_query.fields)
        
        # Score basé sur les agrégations
        complexity_score += len(self.aggregation_types) * 2
        
        # Score basé sur la pagination
        if self.offset > 1000:
            complexity_score += 3
        
        if complexity_score <= 3:
            return "simple"
        elif complexity_score <= 8:
            return "medium"
        else:
            return "complex"


class RequestTransformer:
    """Transformateur de contrats externes vers requêtes internes"""
    
    @staticmethod
    def from_service_contract(contract: SearchServiceQuery) -> InternalSearchRequest:
        """Convertit un contrat externe en requête interne"""
        
        # Déterminer le type de requête interne
        internal_type = RequestTransformer._determine_internal_type(contract)
        
        # Convertir les filtres
        term_filters = RequestTransformer._convert_filters(contract.filters)
        
        # Convertir la recherche textuelle
        text_query = RequestTransformer._convert_text_search(contract.text_search)
        
        # Déterminer le mode de traitement
        processing_mode = RequestTransformer._determine_processing_mode(contract)
        
        # Convertir les agrégations
        agg_fields, agg_types = RequestTransformer._convert_aggregations(contract.aggregations)
        
        # Déterminer la stratégie de cache
        cache_strategy = RequestTransformer._determine_cache_strategy(contract)
        
        return InternalSearchRequest(
            request_id=contract.query_metadata.query_id,
            user_id=contract.query_metadata.user_id,
            query_type=internal_type,
            processing_mode=processing_mode,
            term_filters=term_filters,
            text_query=text_query,
            limit=contract.search_parameters.limit,
            offset=contract.search_parameters.offset,
            min_score=contract.options.min_score,
            aggregation_fields=agg_fields,
            aggregation_types=agg_types,
            include_highlights=contract.options.include_highlights,
            cache_strategy=cache_strategy,
            timeout_ms=contract.search_parameters.timeout_ms,
            original_query_id=contract.query_metadata.query_id,
            agent_context={
                "agent_name": contract.query_metadata.agent_name,
                "team_name": contract.query_metadata.team_name or "",
                "intent_type": contract.query_metadata.intent_type
            }
        )
    
    @staticmethod
    def _determine_internal_type(contract: SearchServiceQuery) -> InternalQueryType:
        """Détermine le type de requête interne"""
        query_type = contract.search_parameters.query_type
        has_text = contract.text_search is not None
        has_filters = len(contract.filters.required) > 1  # > 1 car user_id toujours présent
        has_aggregations = contract.aggregations and contract.aggregations.enabled
        
        if has_aggregations and not has_text and not has_filters:
            return InternalQueryType.AGGREGATION_ONLY
        elif has_text and has_filters:
            return InternalQueryType.HYBRID_SEARCH
        elif has_text and not has_filters:
            return InternalQueryType.TEXT_MATCH
        elif has_filters and len(contract.filters.required) == 2:  # user_id + 1 autre
            return InternalQueryType.SIMPLE_TERM
        elif has_filters:
            return InternalQueryType.MULTI_TERM
        else:
            return InternalQueryType.COMPLEX_QUERY
    
    @staticmethod
    def _convert_filters(filters) -> List[TermFilter]:
        """Convertit les filtres du contrat"""
        term_filters = []
        
        # Convertir les filtres required
        for filter_obj in filters.required:
            term_filters.append(TermFilter(
                field=filter_obj.field,
                value=filter_obj.value,
                operator=filter_obj.operator
            ))
        
        # Convertir les filtres ranges
        for filter_obj in filters.ranges:
            term_filters.append(TermFilter(
                field=filter_obj.field,
                value=filter_obj.value,
                operator=filter_obj.operator
            ))
        
        return term_filters
    
    @staticmethod
    def _convert_text_search(text_config) -> Optional[TextQuery]:
        """Convertit la configuration de recherche textuelle"""
        if not text_config:
            return None
        
        # Boosts par défaut pour les champs financiers
        default_boosts = {
            "searchable_text": 2.0,
            "primary_description": 1.5,
            "merchant_name": 1.8,
            "category_name": 1.2
        }
        
        field_boosts = []
        for field in text_config.fields:
            boost = text_config.boost.get(field, default_boosts.get(field, 1.0)) if text_config.boost else default_boosts.get(field, 1.0)
            field_boosts.append(FieldBoost(field=field, boost=boost))
        
        return TextQuery(
            query=text_config.query,
            fields=text_config.fields,
            field_boosts=field_boosts,
            fuzziness=text_config.fuzziness,
            minimum_should_match=text_config.minimum_should_match
        )
    
    @staticmethod
    def _determine_processing_mode(contract: SearchServiceQuery) -> ProcessingMode:
        """Détermine le mode de traitement optimal"""
        # Basé sur le timeout et la complexité
        timeout = contract.search_parameters.timeout_ms
        has_aggregations = contract.aggregations and contract.aggregations.enabled
        
        if timeout <= 2000:
            return ProcessingMode.FAST
        elif timeout >= 8000 or has_aggregations:
            return ProcessingMode.PRECISE
        else:
            return ProcessingMode.BALANCED
    
    @staticmethod
    def _convert_aggregations(agg_config) -> tuple[List[str], List[AggregationType]]:
        """Convertit la configuration d'agrégations"""
        if not agg_config or not agg_config.enabled:
            return [], []
        
        return agg_config.metrics, agg_config.types
    
    @staticmethod
    def _determine_cache_strategy(contract: SearchServiceQuery) -> CacheStrategy:
        """Détermine la stratégie de cache optimale"""
        if not contract.options.cache_enabled:
            return CacheStrategy.DISABLED
        
        # Cache agressif pour requêtes simples et fréquentes
        intent_type = contract.query_metadata.intent_type
        
        # Intentions fréquentes → cache agressif
        frequent_intents = [
            "SEARCH_BY_CATEGORY",
            "SEARCH_BY_MERCHANT", 
            "COUNT_OPERATIONS"
        ]
        
        if intent_type in frequent_intents:
            return CacheStrategy.AGGRESSIVE
        
        # Requêtes avec agrégations → cache standard
        if contract.aggregations and contract.aggregations.enabled:
            return CacheStrategy.STANDARD
        
        # Recherche textuelle → cache minimal (plus variable)
        if contract.text_search:
            return CacheStrategy.MINIMAL
        
        return CacheStrategy.STANDARD


# === VALIDATION ET HELPERS ===

class RequestValidator:
    """Validateur pour les requêtes internes"""
    
    @staticmethod
    def validate_request(request: InternalSearchRequest) -> bool:
        """Valide une requête interne"""
        # Vérifier que user_id filter existe
        user_filters = [f for f in request.term_filters if f.field == "user_id"]
        if not user_filters:
            raise ValueError("Filtre user_id manquant")
        
        # Vérifier cohérence user_id
        if user_filters[0].value != request.user_id:
            raise ValueError("user_id incohérent entre metadata et filtres")
        
        # Vérifier limites
        if request.limit > 100:
            raise ValueError("limit ne peut pas dépasser 100")
        
        # Vérifier timeout
        if request.timeout_ms > 10000:
            raise ValueError("timeout_ms ne peut pas dépasser 10000")
        
        return True
    
    @staticmethod
    def estimate_performance_impact(request: InternalSearchRequest) -> Dict[str, Any]:
        """Estime l'impact performance d'une requête"""
        impact = {
            "complexity": request.get_estimated_complexity(),
            "expected_time_ms": 0,
            "cache_efficiency": 0.0,
            "elasticsearch_load": "low"
        }
        
        # Estimation temps basée sur la complexité
        base_time = 20  # 20ms de base
        
        # Ajouter temps pour filtres
        base_time += len(request.term_filters) * 5
        
        # Ajouter temps pour recherche textuelle
        if request.text_query:
            base_time += 30
            base_time += len(request.text_query.fields) * 10
        
        # Ajouter temps pour agrégations
        base_time += len(request.aggregation_types) * 25
        
        # Ajouter temps pour pagination profonde
        if request.offset > 1000:
            base_time += 50
        
        impact["expected_time_ms"] = base_time
        
        # Efficacité cache
        if request.cache_strategy == CacheStrategy.AGGRESSIVE:
            impact["cache_efficiency"] = 0.8
        elif request.cache_strategy == CacheStrategy.STANDARD:
            impact["cache_efficiency"] = 0.5
        elif request.cache_strategy == CacheStrategy.MINIMAL:
            impact["cache_efficiency"] = 0.2
        else:
            impact["cache_efficiency"] = 0.0
        
        # Charge Elasticsearch
        if impact["complexity"] == "complex":
            impact["elasticsearch_load"] = "high"
        elif impact["complexity"] == "medium":
            impact["elasticsearch_load"] = "medium"
        
        return impact


# === EXPORTS ===

__all__ = [
    # Enums
    "InternalQueryType",
    "ProcessingMode", 
    "CacheStrategy",
    
    # Dataclasses
    "FieldBoost",
    "TermFilter",
    "TextQuery",
    
    # Modèle principal
    "InternalSearchRequest",
    
    # Transformateur
    "RequestTransformer",
    
    # Validateur
    "RequestValidator"
]