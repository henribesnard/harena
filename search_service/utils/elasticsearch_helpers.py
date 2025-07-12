"""
Utilitaires Elasticsearch pour le Search Service
Helpers spécialisés pour construction de requêtes, formatage et optimisations
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from models import (
    InternalSearchRequest,
    TermFilter,
    TextQuery,
    FieldBoost,
    FilterOperator,
    AggregationType
)
from config import settings, FIELD_CONFIGURATIONS

logger = logging.getLogger(__name__)


class ElasticsearchError(Exception):
    """Exception spécialisée pour les erreurs Elasticsearch"""
    
    def __init__(self, message: str, es_error: Optional[Dict] = None, status_code: Optional[int] = None):
        self.message = message
        self.es_error = es_error
        self.status_code = status_code
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'erreur en dictionnaire pour logs"""
        return {
            "error_type": "elasticsearch_error",
            "message": self.message,
            "status_code": self.status_code,
            "elasticsearch_details": self.es_error
        }


class QueryOptimization(str, Enum):
    """Types d'optimisations appliquées aux requêtes"""
    USER_FILTER_FIRST = "user_filter_first"
    BOOST_OPTIMIZATION = "boost_optimization"
    FIELD_FILTERING = "field_filtering"
    QUERY_SIMPLIFICATION = "query_simplification"
    CACHE_FRIENDLY = "cache_friendly"
    AGGREGATION_OPTIMIZATION = "aggregation_optimization"


@dataclass
class ElasticsearchResponse:
    """Réponse Elasticsearch formatée"""
    took: int
    total_hits: int
    max_score: Optional[float]
    hits: List[Dict[str, Any]]
    aggregations: Optional[Dict[str, Any]] = None
    timed_out: bool = False
    shards: Optional[Dict[str, Any]] = None


class QueryBuilder:
    """Constructeur de requêtes Elasticsearch optimisées"""
    
    def __init__(self):
        self.optimizations_applied = []
    
    def build_from_internal_request(self, request: InternalSearchRequest) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch à partir d'une InternalSearchRequest
        
        Args:
            request: Requête interne validée
            
        Returns:
            Dict: Requête Elasticsearch complète
        """
        self.optimizations_applied.clear()
        
        # Structure de base
        query_body = {
            "query": self._build_query_clause(request),
            "size": request.limit,
            "from": request.offset,
            "_source": self._determine_source_fields(request)
        }
        
        # Ajouter tri optimisé
        query_body["sort"] = self._build_sort_clause(request)
        
        # Ajouter agrégations si demandées
        if request.aggregation_fields or request.aggregation_types:
            query_body["aggs"] = self._build_aggregations(request)
            self.optimizations_applied.append(QueryOptimization.AGGREGATION_OPTIMIZATION)
        
        # Ajouter highlighting si demandé
        if request.include_highlights and request.text_query:
            query_body["highlight"] = self._build_highlight_config(request.text_query)
        
        # Optimisations de performance
        query_body = self._apply_performance_optimizations(query_body, request)
        
        logger.debug(f"Built ES query with optimizations: {self.optimizations_applied}")
        return query_body
    
    def _build_query_clause(self, request: InternalSearchRequest) -> Dict[str, Any]:
        """Construit la clause query principale"""
        bool_query = {
            "bool": {
                "must": [],
                "filter": [],
                "should": []
            }
        }
        
        # 1. Filtres terme (toujours en filter pour performance)
        for term_filter in request.term_filters:
            es_filter = self._convert_term_filter(term_filter)
            
            # user_id en premier pour optimisation
            if term_filter.field == "user_id":
                bool_query["bool"]["filter"].insert(0, es_filter)
                self.optimizations_applied.append(QueryOptimization.USER_FILTER_FIRST)
            else:
                bool_query["bool"]["filter"].append(es_filter)
        
        # 2. Recherche textuelle si présente
        if request.text_query:
            text_queries = self._build_text_search_queries(request.text_query)
            bool_query["bool"]["should"].extend(text_queries)
            
            # Au moins une correspondance textuelle requise
            if text_queries:
                bool_query["bool"]["minimum_should_match"] = 1
        
        # Optimisation: si pas de should, simplifier la structure
        if not bool_query["bool"]["should"]:
            del bool_query["bool"]["should"]
            if "minimum_should_match" in bool_query["bool"]:
                del bool_query["bool"]["minimum_should_match"]
            self.optimizations_applied.append(QueryOptimization.QUERY_SIMPLIFICATION)
        
        # Si pas de must, supprimer
        if not bool_query["bool"]["must"]:
            del bool_query["bool"]["must"]
        
        return bool_query
    
    def _convert_term_filter(self, term_filter: TermFilter) -> Dict[str, Any]:
        """Convertit un TermFilter en filtre Elasticsearch"""
        field = term_filter.field
        value = term_filter.value
        operator = term_filter.operator
        boost = term_filter.boost
        
        if operator == FilterOperator.EQ:
            if boost != 1.0:
                return {"term": {field: {"value": value, "boost": boost}}}
            else:
                return {"term": {field: value}}
        
        elif operator == FilterOperator.IN:
            return {"terms": {field: value}}
        
        elif operator == FilterOperator.NOT_IN:
            return {"bool": {"must_not": {"terms": {field: value}}}}
        
        elif operator == FilterOperator.GT:
            return {"range": {field: {"gt": value}}}
        
        elif operator == FilterOperator.GTE:
            return {"range": {field: {"gte": value}}}
        
        elif operator == FilterOperator.LT:
            return {"range": {field: {"lt": value}}}
        
        elif operator == FilterOperator.LTE:
            return {"range": {field: {"lte": value}}}
        
        elif operator == FilterOperator.BETWEEN:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"BETWEEN operator requires list of 2 values, got {value}")
            return {"range": {field: {"gte": value[0], "lte": value[1]}}}
        
        elif operator == FilterOperator.EXISTS:
            if value:
                return {"exists": {"field": field}}
            else:
                return {"bool": {"must_not": {"exists": {"field": field}}}}
        
        elif operator == FilterOperator.PREFIX:
            return {"prefix": {field: {"value": value, "boost": boost}}}
        
        else:
            raise ValueError(f"Unsupported filter operator: {operator}")
    
    def _build_text_search_queries(self, text_query: TextQuery) -> List[Dict[str, Any]]:
        """Construit les requêtes de recherche textuelle optimisées"""
        queries = []
        query_text = text_query.query.strip()
        
        if not query_text:
            return queries
        
        # 1. Multi-match principal avec boost par champ
        boosted_fields = []
        for field in text_query.fields:
            boost = self._get_field_boost(field, text_query.field_boosts)
            if boost != 1.0:
                boosted_fields.append(f"{field}^{boost}")
            else:
                boosted_fields.append(field)
        
        multi_match = {
            "multi_match": {
                "query": query_text,
                "fields": boosted_fields,
                "type": "best_fields",
                "operator": "or"
            }
        }
        
        # Ajouter fuzziness si configuré
        if text_query.fuzziness:
            multi_match["multi_match"]["fuzziness"] = text_query.fuzziness
        
        # Ajouter minimum_should_match si configuré
        if text_query.minimum_should_match:
            multi_match["multi_match"]["minimum_should_match"] = text_query.minimum_should_match
        
        queries.append(multi_match)
        
        # 2. Match phrase pour correspondances exactes (boost élevé)
        phrase_query = {
            "multi_match": {
                "query": query_text,
                "fields": boosted_fields,
                "type": "phrase",
                "boost": 2.0
            }
        }
        queries.append(phrase_query)
        
        # 3. Match phrase prefix pour autocomplétion
        if len(query_text.split()) <= 3:  # Éviter sur requêtes trop longues
            prefix_query = {
                "multi_match": {
                    "query": query_text,
                    "fields": boosted_fields,
                    "type": "phrase_prefix",
                    "boost": 1.5
                }
            }
            queries.append(prefix_query)
        
        self.optimizations_applied.append(QueryOptimization.BOOST_OPTIMIZATION)
        return queries
    
    def _get_field_boost(self, field: str, field_boosts: List[FieldBoost]) -> float:
        """Récupère le boost pour un champ"""
        for field_boost in field_boosts:
            if field_boost.field == field:
                return field_boost.boost
        
        # Boost par défaut selon la configuration
        if field in FIELD_CONFIGURATIONS:
            return FIELD_CONFIGURATIONS[field].boost
        
        return 1.0
    
    def _determine_source_fields(self, request: InternalSearchRequest) -> Union[bool, List[str]]:
        """Détermine les champs à retourner"""
        # Si pas de champs spécifiés, retourner les champs par défaut
        if not hasattr(request, 'return_fields') or not request.return_fields:
            default_fields = [
                "transaction_id", "user_id", "account_id",
                "amount", "amount_abs", "transaction_type", "currency_code",
                "date", "primary_description", "merchant_name", "category_name",
                "operation_type", "month_year", "weekday"
            ]
            self.optimizations_applied.append(QueryOptimization.FIELD_FILTERING)
            return default_fields
        
        return request.return_fields
    
    def _build_sort_clause(self, request: InternalSearchRequest) -> List[Dict[str, Any]]:
        """Construit la clause de tri optimisée"""
        sort_clause = []
        
        # Si recherche textuelle, trier par score d'abord
        if request.text_query:
            sort_clause.append({"_score": {"order": "desc"}})
        
        # Tri par date décroissante (transactions récentes d'abord)
        sort_clause.append({
            "date": {
                "order": "desc",
                "unmapped_type": "date",
                "missing": "_last"
            }
        })
        
        # Tri de stabilité par transaction_id
        sort_clause.append({
            "transaction_id.keyword": {
                "order": "desc",
                "unmapped_type": "keyword"
            }
        })
        
        return sort_clause
    
    def _build_aggregations(self, request: InternalSearchRequest) -> Dict[str, Any]:
        """Construit les agrégations demandées"""
        aggs = {}
        
        # Agrégations par champ
        for field in request.aggregation_fields:
            if field in ["category_name", "merchant_name"]:
                # Utiliser le champ .keyword pour agrégation
                agg_field = f"{field}.keyword"
                aggs[f"by_{field}"] = {
                    "terms": {
                        "field": agg_field,
                        "size": 20,
                        "order": {"total_amount": "desc"}
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "avg_amount": {"avg": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
            elif field == "month_year":
                aggs["by_month"] = {
                    "terms": {
                        "field": "month_year",
                        "size": 24,  # 2 ans de données
                        "order": {"_key": "desc"}
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
        
        # Agrégations par type
        for agg_type in request.aggregation_types:
            if agg_type == AggregationType.SUM:
                aggs["total_sum"] = {"sum": {"field": "amount_abs"}}
            elif agg_type == AggregationType.AVG:
                aggs["average_amount"] = {"avg": {"field": "amount_abs"}}
            elif agg_type == AggregationType.COUNT:
                aggs["total_count"] = {"value_count": {"field": "transaction_id"}}
            elif agg_type == AggregationType.STATS:
                aggs["amount_stats"] = {"stats": {"field": "amount_abs"}}
            elif agg_type == AggregationType.MIN:
                aggs["min_amount"] = {"min": {"field": "amount_abs"}}
            elif agg_type == AggregationType.MAX:
                aggs["max_amount"] = {"max": {"field": "amount_abs"}}
        
        return aggs
    
    def _build_highlight_config(self, text_query: TextQuery) -> Dict[str, Any]:
        """Construit la configuration de highlighting"""
        highlight_fields = {}
        
        for field in text_query.fields:
            if field in ["searchable_text", "primary_description"]:
                highlight_fields[field] = {
                    "fragment_size": 150,
                    "number_of_fragments": 2,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"]
                }
            elif field == "merchant_name":
                highlight_fields[field] = {
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "number_of_fragments": 0  # Highlight tout le champ
                }
        
        return {
            "fields": highlight_fields,
            "require_field_match": False
        }
    
    def _apply_performance_optimizations(
        self, 
        query_body: Dict[str, Any], 
        request: InternalSearchRequest
    ) -> Dict[str, Any]:
        """Applique les optimisations de performance"""
        
        # Optimisation cache
        if request.cache_strategy.value in ["aggressive", "standard"]:
            query_body["request_cache"] = True
            self.optimizations_applied.append(QueryOptimization.CACHE_FRIENDLY)
        
        # Optimisation pour requêtes simples
        if request.query_type.value == "simple_term":
            # Utiliser constant_score pour les filtres simples
            if "query" in query_body and "bool" in query_body["query"]:
                bool_query = query_body["query"]["bool"]
                if not bool_query.get("should") and not bool_query.get("must"):
                    # Convertir en constant_score pour performance
                    query_body["query"] = {
                        "constant_score": {
                            "filter": bool_query,
                            "boost": 1.0
                        }
                    }
                    self.optimizations_applied.append(QueryOptimization.QUERY_SIMPLIFICATION)
        
        # Timeout adaptatif selon la complexité
        if request.timeout_ms:
            query_body["timeout"] = f"{request.timeout_ms}ms"
        
        return query_body
    
    def get_applied_optimizations(self) -> List[str]:
        """Retourne la liste des optimisations appliquées"""
        return [opt.value for opt in self.optimizations_applied]


class ResponseFormatter:
    """Formateur de réponses Elasticsearch"""
    
    @staticmethod
    def format_elasticsearch_response(es_response: Dict[str, Any]) -> ElasticsearchResponse:
        """
        Formate une réponse Elasticsearch brute en structure standardisée
        
        Args:
            es_response: Réponse brute d'Elasticsearch
            
        Returns:
            ElasticsearchResponse: Structure formatée
        """
        hits_data = es_response.get("hits", {})
        
        # Extraction des hits
        hits = []
        for hit in hits_data.get("hits", []):
            formatted_hit = {
                **hit.get("_source", {}),
                "_score": hit.get("_score"),
                "_id": hit.get("_id")
            }
            
            # Ajouter highlights si présents
            if "highlight" in hit:
                formatted_hit["_highlights"] = hit["highlight"]
            
            hits.append(formatted_hit)
        
        # Extraction du total
        total = hits_data.get("total", {})
        if isinstance(total, dict):
            total_hits = total.get("value", 0)
        else:
            total_hits = total
        
        # Extraction des agrégations
        aggregations = ResponseFormatter._format_aggregations(
            es_response.get("aggregations", {})
        ) if "aggregations" in es_response else None
        
        return ElasticsearchResponse(
            took=es_response.get("took", 0),
            total_hits=total_hits,
            max_score=hits_data.get("max_score"),
            hits=hits,
            aggregations=aggregations,
            timed_out=es_response.get("timed_out", False),
            shards=es_response.get("_shards")
        )
    
    @staticmethod
    def _format_aggregations(aggs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Formate les agrégations Elasticsearch"""
        formatted_aggs = {}
        
        for agg_name, agg_data in aggs_data.items():
            if "buckets" in agg_data:
                # Agrégation terms/histogram
                formatted_aggs[agg_name] = {
                    "buckets": [
                        {
                            "key": bucket.get("key"),
                            "doc_count": bucket.get("doc_count", 0),
                            **{
                                sub_agg_name: sub_agg_data.get("value", sub_agg_data)
                                for sub_agg_name, sub_agg_data in bucket.items()
                                if sub_agg_name not in ["key", "doc_count"]
                            }
                        }
                        for bucket in agg_data["buckets"]
                    ]
                }
            elif "value" in agg_data:
                # Agrégation métrique simple
                formatted_aggs[agg_name] = agg_data["value"]
            else:
                # Agrégation stats ou complexe
                formatted_aggs[agg_name] = agg_data
        
        return formatted_aggs
    
    @staticmethod
    def extract_suggestions(es_response: Dict[str, Any]) -> List[str]:
        """Extrait des suggestions depuis une réponse Elasticsearch"""
        suggestions = []
        
        # Depuis les agrégations terms
        aggs = es_response.get("aggregations", {})
        for agg_name, agg_data in aggs.items():
            if "buckets" in agg_data:
                for bucket in agg_data["buckets"][:5]:  # Top 5
                    key = bucket.get("key")
                    if key and isinstance(key, str):
                        suggestions.append(key)
        
        return list(set(suggestions))  # Dédoublonner


class ErrorHandler:
    """Gestionnaire d'erreurs Elasticsearch spécialisé"""
    
    @staticmethod
    def parse_elasticsearch_error(error_response: Dict[str, Any], status_code: int) -> ElasticsearchError:
        """
        Parse une erreur Elasticsearch et la convertit en exception typée
        
        Args:
            error_response: Réponse d'erreur ES
            status_code: Code de statut HTTP
            
        Returns:
            ElasticsearchError: Exception formatée
        """
        error_type = error_response.get("error", {}).get("type", "unknown")
        error_reason = error_response.get("error", {}).get("reason", "Unknown error")
        
        # Messages d'erreur adaptés selon le type
        if error_type == "index_not_found_exception":
            message = f"Index not found: {error_reason}"
        elif error_type == "parsing_exception":
            message = f"Query parsing error: {error_reason}"
        elif error_type == "search_phase_execution_exception":
            message = f"Search execution error: {error_reason}"
        elif error_type == "timeout_exception":
            message = f"Query timeout: {error_reason}"
        elif status_code == 429:
            message = "Rate limit exceeded - too many requests"
        elif status_code >= 500:
            message = f"Elasticsearch server error: {error_reason}"
        else:
            message = f"Elasticsearch error ({error_type}): {error_reason}"
        
        return ElasticsearchError(
            message=message,
            es_error=error_response,
            status_code=status_code
        )
    
    @staticmethod
    def is_retryable_error(error: ElasticsearchError) -> bool:
        """Détermine si une erreur Elasticsearch peut être retentée"""
        if not error.status_code:
            return False
        
        # Erreurs retryables
        retryable_codes = [429, 502, 503, 504]  # Rate limit, Bad Gateway, Service Unavailable, Gateway Timeout
        retryable_types = ["timeout_exception", "es_rejected_execution_exception"]
        
        if error.status_code in retryable_codes:
            return True
        
        if error.es_error:
            error_type = error.es_error.get("error", {}).get("type", "")
            if error_type in retryable_types:
                return True
        
        return False


class IndexManager:
    """Gestionnaire d'index Elasticsearch"""
    
    @staticmethod
    def get_optimal_index_settings() -> Dict[str, Any]:
        """Retourne les settings optimaux pour l'index de transactions"""
        return {
            "settings": {
                "number_of_shards": 1,  # Single shard pour performance sur dataset moyen
                "number_of_replicas": 0,  # Pas de réplica sur Bonsai basique
                "refresh_interval": "30s",  # Refresh moins fréquent pour performance
                "index": {
                    "max_result_window": settings.max_pagination_offset,
                    "mapping": {
                        "total_fields": {
                            "limit": 50  # Limite raisonnable pour transactions
                        }
                    }
                }
            }
        }
    
    @staticmethod
    def get_transaction_mapping() -> Dict[str, Any]:
        """Retourne le mapping optimisé pour les transactions financières"""
        return {
            "mappings": {
                "properties": {
                    # Champs d'identification
                    "transaction_id": {"type": "keyword"},
                    "user_id": {"type": "integer"},
                    "account_id": {"type": "integer"},
                    
                    # Champs financiers
                    "amount": {"type": "double"},
                    "amount_abs": {"type": "double"},
                    "transaction_type": {"type": "keyword"},
                    "currency_code": {"type": "keyword"},
                    
                    # Champs temporels
                    "date": {"type": "date", "format": "yyyy-MM-dd"},
                    "month_year": {"type": "keyword"},
                    "weekday": {"type": "keyword"},
                    
                    # Champs de recherche textuelle
                    "searchable_text": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard"
                    },
                    "primary_description": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "merchant_name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "category_name": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 100}
                        }
                    },
                    
                    # Champs opérationnels
                    "operation_type": {
                        "type": "keyword",
                        "fields": {
                            "text": {"type": "text"}
                        }
                    }
                }
            }
        }


class QueryOptimizer:
    """Optimiseur de requêtes Elasticsearch"""
    
    @staticmethod
    def optimize_for_cache(query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise une requête pour le cache"""
        # Normaliser l'ordre des clauses pour améliorer le cache hit
        if "query" in query_body and "bool" in query_body["query"]:
            bool_query = query_body["query"]["bool"]
            
            # Trier les filtres pour consistance
            if "filter" in bool_query:
                bool_query["filter"] = sorted(
                    bool_query["filter"],
                    key=lambda x: str(x)
                )
        
        # Activer le cache de requête
        query_body["request_cache"] = True
        
        return query_body
    
    @staticmethod
    def add_track_total_hits(query_body: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
        """Ajoute track_total_hits pour optimiser le comptage"""
        if limit and limit <= 10000:
            # Pour les petites limites, tracker le total exact
            query_body["track_total_hits"] = True
        else:
            # Pour les grandes limites, limiter le tracking pour performance
            query_body["track_total_hits"] = 10000
        
        return query_body
    
    @staticmethod
    def optimize_aggregation_query(query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise une requête avec agrégations"""
        # Ne pas retourner de hits si on fait seulement des agrégations
        if "aggs" in query_body and query_body.get("size", 10) > 0:
            # Si le focus est sur les agrégations, réduire les hits
            query_body["size"] = 0
        
        return query_body


# === FONCTIONS UTILITAIRES ===

def build_simple_user_query(user_id: int, limit: int = 20) -> Dict[str, Any]:
    """Construit une requête simple pour un utilisateur"""
    return {
        "query": {
            "term": {"user_id": user_id}
        },
        "size": limit,
        "sort": [{"date": {"order": "desc"}}]
    }


def build_text_search_query(
    user_id: int, 
    query_text: str, 
    fields: Optional[List[str]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Construit une requête de recherche textuelle simple"""
    search_fields = fields or ["searchable_text", "primary_description", "merchant_name"]
    
    return {
        "query": {
            "bool": {
                "filter": [{"term": {"user_id": user_id}}],
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": search_fields,
                            "type": "best_fields",
                            "operator": "or"
                        }
                    }
                ]
            }
        },
        "size": limit,
        "sort": [
            {"_score": {"order": "desc"}},
            {"date": {"order": "desc"}}
        ]
    }


def validate_elasticsearch_response(response: Dict[str, Any]) -> bool:
    """Valide qu'une réponse Elasticsearch est bien formée"""
    required_fields = ["took", "hits"]
    
    for field in required_fields:
        if field not in response:
            return False
    
    if "hits" not in response.get("hits", {}):
        return False
    
    return True


def extract_error_details(es_error: Dict[str, Any]) -> Dict[str, str]:
    """Extrait les détails utiles d'une erreur Elasticsearch"""
    error_info = es_error.get("error", {})
    
    return {
        "type": error_info.get("type", "unknown"),
        "reason": error_info.get("reason", "Unknown error"),
        "index": error_info.get("index", "unknown"),
        "caused_by": error_info.get("caused_by", {}).get("reason", "")
    }


def estimate_query_cost(query_body: Dict[str, Any]) -> str:
    """Estime le coût d'une requête (low/medium/high)"""
    cost_score = 0
    
    # Coût basé sur la taille
    size = query_body.get("size", 10)
    cost_score += min(size // 20, 3)
    
    # Coût basé sur les agrégations
    if "aggs" in query_body:
        cost_score += len(query_body["aggs"]) * 2
    
    # Coût basé sur la complexité de la requête
    query = query_body.get("query", {})
    if "bool" in query:
        bool_query = query["bool"]
        cost_score += len(bool_query.get("should", []))
        cost_score += len(bool_query.get("must", []))
        cost_score += len(bool_query.get("filter", []))
    
    # Coût basé sur les highlights
    if "highlight" in query_body:
        cost_score += 1
    
    # Coût basé sur la pagination
    from_offset = query_body.get("from", 0)
    if from_offset > 1000:
        cost_score += 3
    elif from_offset > 100:
        cost_score += 1
    
    # Classification finale
    if cost_score <= 3:
        return "low"
    elif cost_score <= 8:
        return "medium"
    else:
        return "high"


def normalize_query_for_cache(query_body: Dict[str, Any]) -> str:
    """Normalise une requête pour génération de clé de cache stable"""
    import json
    import hashlib
    
    # Créer une copie pour normalisation
    normalized = dict(query_body)
    
    # Supprimer les champs qui ne doivent pas affecter le cache
    cache_excluded_fields = ["from", "size", "timeout", "_source"]
    for field in cache_excluded_fields:
        normalized.pop(field, None)
    
    # Normaliser les booléens dans bool queries
    if "query" in normalized and "bool" in normalized["query"]:
        bool_query = normalized["query"]["bool"]
        for clause in ["must", "filter", "should", "must_not"]:
            if clause in bool_query:
                # Trier pour ordre stable
                bool_query[clause] = sorted(bool_query[clause], key=lambda x: json.dumps(x, sort_keys=True))
    
    # Créer hash stable
    normalized_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(normalized_str.encode()).hexdigest()


def build_count_query(base_query: Dict[str, Any]) -> Dict[str, Any]:
    """Convertit une requête search en requête count optimisée"""
    count_query = {"query": base_query.get("query", {"match_all": {}})}
    
    # Supprimer les éléments non nécessaires pour count
    excluded_for_count = ["sort", "highlight", "_source", "size", "from"]
    
    return count_query


def merge_query_filters(base_filters: List[Dict[str, Any]], additional_filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fusionne des listes de filtres en évitant les doublons"""
    merged = list(base_filters)
    
    for new_filter in additional_filters:
        # Vérifier si un filtre similaire existe déjà
        is_duplicate = False
        for existing_filter in merged:
            if _are_filters_equivalent(existing_filter, new_filter):
                is_duplicate = True
                break
        
        if not is_duplicate:
            merged.append(new_filter)
    
    return merged


def _are_filters_equivalent(filter1: Dict[str, Any], filter2: Dict[str, Any]) -> bool:
    """Compare deux filtres pour détecter les équivalences"""
    import json
    # Comparaison simple basée sur la structure
    return json.dumps(filter1, sort_keys=True) == json.dumps(filter2, sort_keys=True)


def get_field_mapping_type(field_name: str) -> Optional[str]:
    """Retourne le type de mapping Elasticsearch pour un champ"""
    mapping = IndexManager.get_transaction_mapping()
    properties = mapping.get("mappings", {}).get("properties", {})
    
    if field_name in properties:
        return properties[field_name].get("type")
    
    # Vérifier les champs avec sous-champs (.keyword)
    if "." in field_name:
        base_field, sub_field = field_name.split(".", 1)
        if base_field in properties:
            fields = properties[base_field].get("fields", {})
            if sub_field in fields:
                return fields[sub_field].get("type")
    
    return None


def optimize_pagination(query_body: Dict[str, Any], from_offset: int, size: int) -> Dict[str, Any]:
    """Optimise la pagination selon l'offset"""
    query_body["from"] = from_offset
    query_body["size"] = size
    
    # Pour les offsets élevés, suggérer search_after
    if from_offset > 5000:
        logger.warning(f"High pagination offset ({from_offset}). Consider using search_after for better performance.")
        
        # Ajouter track_total_hits pour limiter le coût
        query_body["track_total_hits"] = 10000
    
    return query_body


class FieldAnalyzer:
    """Analyseur de champs pour optimisation des requêtes"""
    
    @staticmethod
    def get_searchable_fields() -> List[str]:
        """Retourne la liste des champs searchables"""
        searchable = []
        for field, config in FIELD_CONFIGURATIONS.items():
            if config.is_searchable:
                searchable.append(field)
        return searchable
    
    @staticmethod
    def get_filterable_fields() -> List[str]:
        """Retourne la liste des champs filtrables"""
        filterable = []
        for field, config in FIELD_CONFIGURATIONS.items():
            if config.is_filterable:
                filterable.append(field)
        return filterable
    
    @staticmethod
    def get_aggregatable_fields() -> List[str]:
        """Retourne la liste des champs agregables"""
        aggregatable = []
        for field, config in FIELD_CONFIGURATIONS.items():
            if config.supports_aggregation:
                aggregatable.append(field)
        return aggregatable
    
    @staticmethod
    def suggest_field_optimization(field: str, operation: str) -> Optional[str]:
        """Suggère des optimisations pour un champ selon l'opération"""
        if field not in FIELD_CONFIGURATIONS:
            return f"Field {field} not found in configuration"
        
        config = FIELD_CONFIGURATIONS[field]
        
        if operation == "search" and not config.is_searchable:
            return f"Field {field} is not searchable. Consider using filterable fields."
        
        if operation == "filter" and not config.is_filterable:
            return f"Field {field} is not filterable."
        
        if operation == "aggregate" and not config.supports_aggregation:
            # Suggérer le champ .keyword si applicable
            if config.field_type.value == "text":
                return f"Use {field}.keyword for aggregation instead of {field}"
            else:
                return f"Field {field} does not support aggregation"
        
        return None


class QueryPerformanceAnalyzer:
    """Analyseur de performance des requêtes"""
    
    @staticmethod
    def analyze_query_performance(query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les aspects performance d'une requête"""
        analysis = {
            "estimated_cost": estimate_query_cost(query_body),
            "optimizations_suggested": [],
            "warnings": [],
            "complexity_score": 0
        }
        
        # Analyse de la pagination
        from_offset = query_body.get("from", 0)
        size = query_body.get("size", 10)
        
        if from_offset > 1000:
            analysis["warnings"].append("Deep pagination detected - consider search_after")
            analysis["complexity_score"] += 3
        
        if size > 50:
            analysis["warnings"].append("Large page size may impact performance")
            analysis["complexity_score"] += 2
        
        # Analyse des agrégations
        if "aggs" in query_body:
            agg_count = len(query_body["aggs"])
            if agg_count > 3:
                analysis["warnings"].append("Multiple aggregations may be slow")
                analysis["complexity_score"] += 2
            
            # Vérifier les agrégations sur des champs text (non-optimaux)
            for agg_name, agg_config in query_body["aggs"].items():
                if "terms" in agg_config:
                    field = agg_config["terms"].get("field", "")
                    if not field.endswith(".keyword") and get_field_mapping_type(field) == "text":
                        analysis["optimizations_suggested"].append(
                            f"Use {field}.keyword instead of {field} for terms aggregation"
                        )
        
        # Analyse de la requête bool
        query = query_body.get("query", {})
        if "bool" in query:
            bool_query = query["bool"]
            
            # Compter les clauses
            clause_count = (
                len(bool_query.get("must", [])) +
                len(bool_query.get("should", [])) +
                len(bool_query.get("filter", [])) +
                len(bool_query.get("must_not", []))
            )
            
            analysis["complexity_score"] += clause_count
            
            if clause_count > 10:
                analysis["warnings"].append("Complex boolean query with many clauses")
            
            # Vérifier l'ordre des filtres
            filters = bool_query.get("filter", [])
            if filters:
                first_filter = filters[0]
                if not (isinstance(first_filter, dict) and 
                       "term" in first_filter and 
                       "user_id" in first_filter.get("term", {})):
                    analysis["optimizations_suggested"].append(
                        "Place user_id filter first for better performance"
                    )
        
        # Analyse des highlights
        if "highlight" in query_body:
            highlight_fields = query_body["highlight"].get("fields", {})
            if len(highlight_fields) > 3:
                analysis["warnings"].append("Many highlight fields may impact performance")
                analysis["complexity_score"] += 1
        
        return analysis
    
    @staticmethod
    def get_optimization_recommendations(analysis: Dict[str, Any]) -> List[str]:
        """Retourne des recommandations d'optimisation basées sur l'analyse"""
        recommendations = []
        
        # Recommandations basées sur le coût
        if analysis["estimated_cost"] == "high":
            recommendations.append("Consider simplifying the query or reducing result size")
        
        # Recommandations basées sur la complexité
        if analysis["complexity_score"] > 15:
            recommendations.append("Query is very complex - consider breaking into multiple simpler queries")
        elif analysis["complexity_score"] > 8:
            recommendations.append("Consider caching results for this complex query")
        
        # Ajouter les optimisations suggérées
        recommendations.extend(analysis.get("optimizations_suggested", []))
        
        return recommendations


# === HELPERS DE DEBUGGING ===

def explain_query_execution(query_body: Dict[str, Any]) -> Dict[str, Any]:
    """Ajoute explain=true à une requête pour debugging"""
    debug_query = dict(query_body)
    debug_query["explain"] = True
    return debug_query


def profile_query_execution(query_body: Dict[str, Any]) -> Dict[str, Any]:
    """Ajoute profiling à une requête pour analyse détaillée"""
    debug_query = dict(query_body)
    debug_query["profile"] = True
    return debug_query


def log_query_for_debug(query_body: Dict[str, Any], context: str = ""):
    """Log une requête pour debugging avec formatting"""
    import json
    logger.debug(f"Elasticsearch query {context}:")
    logger.debug(json.dumps(query_body, indent=2, ensure_ascii=False))


# === EXPORTS ===

__all__ = [
    # Exceptions
    "ElasticsearchError",
    
    # Classes principales
    "QueryBuilder",
    "ResponseFormatter", 
    "ErrorHandler",
    "IndexManager",
    "QueryOptimizer",
    "FieldAnalyzer",
    "QueryPerformanceAnalyzer",
    
    # Structures de données
    "ElasticsearchResponse",
    "QueryOptimization",
    
    # Fonctions utilitaires
    "build_simple_user_query",
    "build_text_search_query",
    "validate_elasticsearch_response",
    "extract_error_details",
    "estimate_query_cost",
    "normalize_query_for_cache",
    "build_count_query",
    "merge_query_filters",
    "get_field_mapping_type",
    "optimize_pagination",
    
    # Debugging
    "explain_query_execution",
    "profile_query_execution", 
    "log_query_for_debug"
]