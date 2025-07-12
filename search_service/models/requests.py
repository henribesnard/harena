"""
Modèles de requêtes internes du Search Service
==============================================

Structures simplifiées pour le traitement interne des contrats avec transformateurs
bidirectionnels pour convertir entre les contrats externes et les modèles internes.

Classes principales :
- InternalSearchRequest : Requête interne optimisée
- RequestTransformer : Conversions contrat ↔ requête interne
- ResponseTransformer : Conversions réponse interne ↔ contrat
- ValidationRequest/TemplateRequest : Modèles API spécialisés

Architecture :
    SearchServiceQuery → RequestTransformer.from_contract() → InternalSearchRequest
    InternalSearchResponse → ResponseTransformer.to_contract() → SearchServiceResponse
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass
from datetime import datetime, timezone

from .service_contracts import (
    FilterOperator, 
    AggregationType,
    SearchServiceQuery,
    SearchServiceResponse
)


# === ENUMS INTERNES ===

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


# === STRUCTURES DE DONNÉES INTERNES ===

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


# === MODÈLE PRINCIPAL INTERNE ===

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


# === TRANSFORMATEUR PRINCIPAL ===

class RequestTransformer:
    """Transformateur de contrats externes vers requêtes internes"""
    
    @staticmethod
    def from_contract(contract: SearchServiceQuery) -> InternalSearchRequest:
        """
        Convertit un contrat externe en requête interne
        MÉTHODE PRINCIPALE utilisée par query_executor.py
        
        Args:
            contract: Contrat SearchServiceQuery du conversation service
            
        Returns:
            InternalSearchRequest: Requête interne optimisée
        """
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
    def from_service_contract(contract: SearchServiceQuery) -> InternalSearchRequest:
        """
        Alias pour from_contract pour compatibilité
        MÉTHODE ALTERNATIVE utilisée si nécessaire
        """
        return RequestTransformer.from_contract(contract)
    
    @staticmethod
    def to_contract(internal_request: InternalSearchRequest) -> SearchServiceQuery:
        """
        Convertit une requête interne en contrat externe
        MÉTHODE INVERSE (rarement utilisée mais nécessaire pour certains cas)
        
        Args:
            internal_request: Requête interne
            
        Returns:
            SearchServiceQuery: Contrat externe reconstruit
        """
        # Reconstruire les métadonnées
        query_metadata = {
            "query_id": internal_request.request_id,
            "user_id": internal_request.user_id,
            "agent_name": internal_request.agent_context.get("agent_name", "search_service") if internal_request.agent_context else "search_service",
            "team_name": internal_request.agent_context.get("team_name") if internal_request.agent_context else None,
            "intent_type": internal_request.agent_context.get("intent_type", "MANUAL_SEARCH") if internal_request.agent_context else "MANUAL_SEARCH",
            "original_query": f"Internal query {internal_request.query_type.value}",
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Reconstruire les paramètres de recherche
        search_parameters = {
            "query_type": internal_request.query_type.value,
            "limit": internal_request.limit,
            "offset": internal_request.offset,
            "timeout_ms": internal_request.timeout_ms,
            "fields": ["*"]  # Tous les champs par défaut
        }
        
        # Reconstruire les filtres
        filters = {
            "required": [],
            "ranges": []
        }
        
        for term_filter in internal_request.term_filters:
            filter_obj = {
                "field": term_filter.field,
                "operator": term_filter.operator,
                "value": term_filter.value
            }
            
            if term_filter.operator in [FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE, FilterOperator.BETWEEN]:
                filters["ranges"].append(filter_obj)
            else:
                filters["required"].append(filter_obj)
        
        # Reconstruire la recherche textuelle
        text_search = None
        if internal_request.text_query:
            text_search = {
                "query": internal_request.text_query.query,
                "fields": internal_request.text_query.fields,
                "boost": {fb.field: fb.boost for fb in internal_request.text_query.field_boosts},
                "fuzziness": internal_request.text_query.fuzziness,
                "minimum_should_match": internal_request.text_query.minimum_should_match
            }
        
        # Reconstruire les agrégations
        aggregations = None
        if internal_request.aggregation_fields or internal_request.aggregation_types:
            aggregations = {
                "enabled": True,
                "types": internal_request.aggregation_types,
                "group_by": internal_request.aggregation_fields,
                "metrics": ["amount_abs", "transaction_id"]  # Métriques par défaut
            }
        
        # Reconstruire les options
        options = {
            "include_highlights": internal_request.include_highlights,
            "include_explanation": False,
            "cache_enabled": internal_request.cache_strategy != CacheStrategy.DISABLED,
            "return_raw_elasticsearch": False,
            "min_score": internal_request.min_score
        }
        
        # Créer le contrat reconstruit
        # Note: Ceci nécessite que SearchServiceQuery soit importable et constructible
        try:
            return SearchServiceQuery(
                query_metadata=query_metadata,
                search_parameters=search_parameters,
                filters=filters,
                text_search=text_search,
                aggregations=aggregations,
                options=options
            )
        except Exception as e:
            # Fallback si la construction échoue
            raise ValueError(f"Cannot convert internal request to contract: {e}")
    
    @staticmethod
    def _determine_internal_type(contract: SearchServiceQuery) -> InternalQueryType:
        """Détermine le type de requête interne"""
        has_text = contract.text_search is not None and contract.text_search.query.strip()
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
        if not text_config or not text_config.query.strip():
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
        
        return agg_config.group_by or [], agg_config.types or []
    
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


# === TRANSFORMATEUR DE RÉPONSES ===

class ResponseTransformer:
    """Transformateur de réponses internes vers contrats externes"""
    
    @staticmethod
    def to_contract(internal_response) -> SearchServiceResponse:
        """
        Convertit une réponse interne en contrat externe
        MÉTHODE PRINCIPALE utilisée par query_executor.py
        
        Args:
            internal_response: Réponse interne (InternalSearchResponse)
            
        Returns:
            SearchServiceResponse: Contrat de réponse externe
        """
        # Importer ici pour éviter les imports circulaires
        try:
            from .responses import InternalSearchResponse
        except ImportError:
            # Fallback si responses.py n'est pas disponible
            pass
        
        # Métadonnées de réponse
        response_metadata = {
            "query_id": getattr(internal_response, 'request_id', 'unknown'),
            "execution_time_ms": getattr(internal_response, 'execution_time_ms', 0),
            "total_hits": getattr(internal_response, 'total_hits', 0),
            "returned_hits": len(getattr(internal_response, 'raw_results', [])),
            "has_more": getattr(internal_response, 'total_hits', 0) > len(getattr(internal_response, 'raw_results', [])),
            "cache_hit": getattr(internal_response, 'served_from_cache', False),
            "elasticsearch_took": getattr(internal_response, 'elasticsearch_took', 0),
            "agent_context": {
                "requesting_agent": "search_service",
                "next_suggested_agent": "response_generator_agent"
            },
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Résultats standardisés
        results = []
        raw_results = getattr(internal_response, 'raw_results', [])
        if hasattr(internal_response, 'get_standardized_results'):
            results = internal_response.get_standardized_results()
        else:
            # Fallback : convertir raw_results si possible
            for result in raw_results:
                if hasattr(result, 'to_standardized_result'):
                    results.append(result.to_standardized_result())
                elif isinstance(result, dict):
                    results.append(result)
        
        # Agrégations
        aggregations = None
        if hasattr(internal_response, 'aggregations') and internal_response.aggregations:
            aggregations = ResponseTransformer._convert_internal_aggregations(internal_response.aggregations)
        
        # Métriques de performance
        performance = {
            "query_complexity": getattr(internal_response, 'complexity', 'medium'),
            "optimization_applied": [],
            "index_used": "harena_transactions",
            "shards_queried": 1,
            "cache_hit": getattr(internal_response, 'served_from_cache', False)
        }
        
        # Enrichissement contextuel
        context_enrichment = {
            "search_intent_matched": True,
            "result_quality_score": getattr(internal_response, 'quality_score', 0.5),
            "suggested_followup_questions": getattr(internal_response, 'suggested_followups', []),
            "next_suggested_agent": "response_generator_agent" if results else None
        }
        
        # Debug (optionnel)
        debug_info = None
        
        # Construire le contrat de réponse
        return SearchServiceResponse(
            response_metadata=response_metadata,
            results=results,
            aggregations=aggregations,
            performance=performance,
            context_enrichment=context_enrichment,
            debug=debug_info
        )
    
    @staticmethod
    def _convert_internal_aggregations(internal_aggs) -> Dict[str, Any]:
        """Convertit les agrégations internes en format contrat"""
        result = {
            "total_amount": 0.0,
            "transaction_count": 0,
            "average_amount": 0.0,
            "by_month": [],
            "by_category": [],
            "by_merchant": [],
            "statistics": {}
        }
        
        # Traiter chaque agrégation interne
        for agg in internal_aggs:
            if hasattr(agg, 'name'):
                name = agg.name
                if name == "monthly" or name == "by_month":
                    result["by_month"] = ResponseTransformer._convert_agg_buckets(agg)
                elif name == "category" or name == "by_category":
                    result["by_category"] = ResponseTransformer._convert_agg_buckets(agg)
                elif name == "merchant" or name == "by_merchant":
                    result["by_merchant"] = ResponseTransformer._convert_agg_buckets(agg)
                
                # Accumulation des totaux
                if hasattr(agg, 'total_amount') and agg.total_amount:
                    result["total_amount"] += agg.total_amount
                if hasattr(agg, 'total_count') and agg.total_count:
                    result["transaction_count"] += agg.total_count
        
        # Calculs finaux
        if result["transaction_count"] > 0:
            result["average_amount"] = result["total_amount"] / result["transaction_count"]
        
        return result
    
    @staticmethod
    def _convert_agg_buckets(agg) -> List[Dict[str, Any]]:
        """Convertit les buckets d'agrégation"""
        if not hasattr(agg, 'buckets'):
            return []
        
        converted_buckets = []
        for bucket in agg.buckets:
            if hasattr(bucket, 'to_external_bucket'):
                converted_buckets.append(bucket.to_external_bucket())
            else:
                # Fallback : structure basique
                converted_buckets.append({
                    "key": getattr(bucket, 'key', 'unknown'),
                    "doc_count": getattr(bucket, 'doc_count', 0),
                    "total_amount": getattr(bucket, 'total_amount', 0.0)
                })
        
        return converted_buckets


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


# === MODÈLES API SPÉCIALISÉS ===

class ValidationRequest(BaseModel):
    """Requête de validation pour l'endpoint /validate"""
    query: Dict[str, Any] = Field(..., description="Requête à valider")
    validate_security: bool = Field(default=True, description="Valider la sécurité")
    validate_performance: bool = Field(default=True, description="Valider la performance")
    validate_syntax: bool = Field(default=True, description="Valider la syntaxe")
    
    @field_validator("query")
    @classmethod
    def validate_query_structure(cls, v):
        """Valide la structure de base de la requête"""
        if not isinstance(v, dict):
            raise ValueError("query doit être un dictionnaire")
        
        # Vérifications de base
        required_fields = ["query_metadata", "search_parameters"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Champ obligatoire manquant: {field}")
        
        return v


class TemplateRequest(BaseModel):
    """Requête de template pour l'endpoint /templates"""
    template_name: Optional[str] = Field(default=None, description="Nom du template spécifique")
    category: Optional[str] = Field(default=None, description="Catégorie de templates")
    intent_type: Optional[str] = Field(default=None, description="Type d'intention")
    include_usage_stats: bool = Field(default=False, description="Inclure les statistiques d'usage")
    
    @field_validator("template_name")
    @classmethod
    def validate_template_name(cls, v):
        """Valide le nom du template"""
        if v is not None and not v.strip():
            raise ValueError("template_name ne peut pas être vide")
        return v


# === UTILITAIRES DE CONVERSION ===

class ContractConverter:
    """Convertisseur entre différents formats de contrats"""
    
    @staticmethod
    def dict_to_internal_request(data: Dict[str, Any]) -> InternalSearchRequest:
        """Convertit un dictionnaire en InternalSearchRequest"""
        
        # Valeurs par défaut
        defaults = {
            "request_id": f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_id": data.get("user_id", 1),
            "query_type": InternalQueryType.SIMPLE_TERM,
            "processing_mode": ProcessingMode.BALANCED,
            "term_filters": [],
            "limit": 20,
            "offset": 0,
            "timeout_ms": 5000,
            "cache_strategy": CacheStrategy.STANDARD
        }
        
        # Merger avec les données fournies
        merged_data = {**defaults, **data}
        
        # Convertir les filtres si nécessaire
        if "filters" in data and isinstance(data["filters"], list):
            term_filters = []
            for f in data["filters"]:
                if isinstance(f, dict):
                    term_filters.append(TermFilter(
                        field=f.get("field", "unknown"),
                        value=f.get("value", ""),
                        operator=FilterOperator(f.get("operator", "eq"))
                    ))
            merged_data["term_filters"] = term_filters
        
        # Convertir la recherche textuelle si nécessaire
        if "text_search" in data and isinstance(data["text_search"], dict):
            text_data = data["text_search"]
            field_boosts = []
            
            # Boosts par défaut
            default_boosts = {
                "searchable_text": 2.0,
                "primary_description": 1.5,
                "merchant_name": 1.8
            }
            
            fields = text_data.get("fields", ["searchable_text", "primary_description"])
            for field in fields:
                boost = text_data.get("boost", {}).get(field, default_boosts.get(field, 1.0))
                field_boosts.append(FieldBoost(field=field, boost=boost))
            
            merged_data["text_query"] = TextQuery(
                query=text_data.get("query", ""),
                fields=fields,
                field_boosts=field_boosts,
                fuzziness=text_data.get("fuzziness"),
                minimum_should_match=text_data.get("minimum_should_match")
            )
        
        return InternalSearchRequest(**merged_data)
    
    @staticmethod
    def internal_request_to_dict(request: InternalSearchRequest) -> Dict[str, Any]:
        """Convertit un InternalSearchRequest en dictionnaire"""
        
        result = {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "query_type": request.query_type.value,
            "processing_mode": request.processing_mode.value,
            "limit": request.limit,
            "offset": request.offset,
            "timeout_ms": request.timeout_ms,
            "cache_strategy": request.cache_strategy.value,
            "include_highlights": request.include_highlights,
            "aggregation_fields": request.aggregation_fields,
            "aggregation_types": [t.value for t in request.aggregation_types]
        }
        
        # Convertir les filtres
        if request.term_filters:
            result["filters"] = [
                {
                    "field": f.field,
                    "value": f.value,
                    "operator": f.operator.value,
                    "boost": f.boost
                }
                for f in request.term_filters
            ]
        
        # Convertir la recherche textuelle
        if request.text_query:
            result["text_search"] = {
                "query": request.text_query.query,
                "fields": request.text_query.fields,
                "boost": {fb.field: fb.boost for fb in request.text_query.field_boosts},
                "fuzziness": request.text_query.fuzziness,
                "minimum_should_match": request.text_query.minimum_should_match
            }
        
        # Contexte agent
        if request.agent_context:
            result["agent_context"] = request.agent_context
        
        return result
    
    @staticmethod
    def elasticsearch_response_to_dict(es_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convertit une réponse Elasticsearch brute en format simplifié"""
        
        hits_data = es_response.get("hits", {})
        
        # Extraction basique des résultats
        results = []
        for hit in hits_data.get("hits", []):
            result = {
                **hit.get("_source", {}),
                "_score": hit.get("_score", 0.0),
                "_id": hit.get("_id", "")
            }
            
            # Ajouter highlights si présents
            if "highlight" in hit:
                result["_highlights"] = hit["highlight"]
            
            results.append(result)
        
        # Total des résultats
        total = hits_data.get("total", {})
        if isinstance(total, dict):
            total_hits = total.get("value", 0)
        else:
            total_hits = total or 0
        
        return {
            "took": es_response.get("took", 0),
            "total_hits": total_hits,
            "max_score": hits_data.get("max_score"),
            "results": results,
            "aggregations": es_response.get("aggregations", {}),
            "timed_out": es_response.get("timed_out", False)
        }


# === FACTORY POUR REQUÊTES TYPES ===

class RequestFactory:
    """Factory pour créer des requêtes types courantes"""
    
    @staticmethod
    def create_simple_search(user_id: int, query: str, limit: int = 20) -> InternalSearchRequest:
        """Crée une requête de recherche simple"""
        
        # Filtres de base
        term_filters = [
            TermFilter(field="user_id", value=user_id, operator=FilterOperator.EQ)
        ]
        
        # Recherche textuelle
        field_boosts = [
            FieldBoost(field="searchable_text", boost=2.0),
            FieldBoost(field="primary_description", boost=1.5),
            FieldBoost(field="merchant_name", boost=1.8)
        ]
        
        text_query = TextQuery(
            query=query,
            fields=["searchable_text", "primary_description", "merchant_name"],
            field_boosts=field_boosts
        )
        
        return InternalSearchRequest(
            request_id=f"simple_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            user_id=user_id,
            query_type=InternalQueryType.TEXT_MATCH,
            processing_mode=ProcessingMode.FAST,
            term_filters=term_filters,
            text_query=text_query,
            limit=limit,
            cache_strategy=CacheStrategy.STANDARD
        )
    
    @staticmethod
    def create_category_filter(user_id: int, category: str, limit: int = 20) -> InternalSearchRequest:
        """Crée une requête de filtrage par catégorie"""
        
        term_filters = [
            TermFilter(field="user_id", value=user_id, operator=FilterOperator.EQ),
            TermFilter(field="category_name", value=category, operator=FilterOperator.EQ)
        ]
        
        return InternalSearchRequest(
            request_id=f"category_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            user_id=user_id,
            query_type=InternalQueryType.SIMPLE_TERM,
            processing_mode=ProcessingMode.FAST,
            term_filters=term_filters,
            limit=limit,
            cache_strategy=CacheStrategy.AGGRESSIVE
        )
    
    @staticmethod
    def create_aggregation_request(user_id: int, group_by: List[str], 
                                 agg_types: List[AggregationType]) -> InternalSearchRequest:
        """Crée une requête d'agrégation"""
        
        term_filters = [
            TermFilter(field="user_id", value=user_id, operator=FilterOperator.EQ)
        ]
        
        return InternalSearchRequest(
            request_id=f"agg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            user_id=user_id,
            query_type=InternalQueryType.AGGREGATION_ONLY,
            processing_mode=ProcessingMode.PRECISE,
            term_filters=term_filters,
            limit=0,  # Pas de résultats individuels pour agrégation pure
            aggregation_fields=group_by,
            aggregation_types=agg_types,
            cache_strategy=CacheStrategy.STANDARD
        )
    
    @staticmethod
    def create_complex_search(user_id: int, query: str, category: Optional[str] = None,
                            amount_min: Optional[float] = None, amount_max: Optional[float] = None,
                            date_from: Optional[str] = None, date_to: Optional[str] = None,
                            limit: int = 20) -> InternalSearchRequest:
        """Crée une requête complexe avec multiple critères"""
        
        # Filtres de base
        term_filters = [
            TermFilter(field="user_id", value=user_id, operator=FilterOperator.EQ)
        ]
        
        # Ajouter filtres selon les paramètres
        if category:
            term_filters.append(
                TermFilter(field="category_name", value=category, operator=FilterOperator.EQ)
            )
        
        if amount_min is not None and amount_max is not None:
            term_filters.append(
                TermFilter(field="amount_abs", value=[amount_min, amount_max], operator=FilterOperator.BETWEEN)
            )
        elif amount_min is not None:
            term_filters.append(
                TermFilter(field="amount_abs", value=amount_min, operator=FilterOperator.GTE)
            )
        elif amount_max is not None:
            term_filters.append(
                TermFilter(field="amount_abs", value=amount_max, operator=FilterOperator.LTE)
            )
        
        if date_from and date_to:
            term_filters.append(
                TermFilter(field="date", value=[date_from, date_to], operator=FilterOperator.BETWEEN)
            )
        
        # Recherche textuelle si fournie
        text_query = None
        if query and query.strip():
            field_boosts = [
                FieldBoost(field="searchable_text", boost=2.0),
                FieldBoost(field="primary_description", boost=1.5),
                FieldBoost(field="merchant_name", boost=1.8)
            ]
            
            text_query = TextQuery(
                query=query,
                fields=["searchable_text", "primary_description", "merchant_name"],
                field_boosts=field_boosts
            )
        
        # Déterminer le type de requête
        has_text = text_query is not None
        has_multiple_filters = len(term_filters) > 2  # > 2 car user_id + au moins 1 autre
        
        if has_text and has_multiple_filters:
            query_type = InternalQueryType.HYBRID_SEARCH
        elif has_text:
            query_type = InternalQueryType.TEXT_MATCH
        elif has_multiple_filters:
            query_type = InternalQueryType.MULTI_TERM
        else:
            query_type = InternalQueryType.COMPLEX_QUERY
        
        return InternalSearchRequest(
            request_id=f"complex_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            user_id=user_id,
            query_type=query_type,
            processing_mode=ProcessingMode.BALANCED,
            term_filters=term_filters,
            text_query=text_query,
            limit=limit,
            cache_strategy=CacheStrategy.MINIMAL,  # Cache minimal pour requêtes complexes
            include_highlights=bool(text_query)
        )


# === EXPORTS ===

__all__ = [
    # === ENUMS ===
    "InternalQueryType",
    "ProcessingMode", 
    "CacheStrategy",
    
    # === STRUCTURES DE DONNÉES ===
    "FieldBoost",
    "TermFilter",
    "TextQuery",
    
    # === MODÈLE PRINCIPAL ===
    "InternalSearchRequest",
    
    # === TRANSFORMATEURS PRINCIPAUX ===
    "RequestTransformer",     # CLASSE PRINCIPALE avec from_contract() et to_contract()
    "ResponseTransformer",    # CLASSE PRINCIPALE avec to_contract()
    
    # === VALIDATION ===
    "RequestValidator",
    
    # === MODÈLES API ===
    "ValidationRequest",
    "TemplateRequest",
    
    # === UTILITAIRES ===
    "ContractConverter",
    "RequestFactory"
]