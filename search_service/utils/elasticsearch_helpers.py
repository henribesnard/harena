"""
Utilitaires Elasticsearch pour le Search Service
===============================================

Helpers spécialisés pour construction de requêtes, formatage et optimisations.
Fournit une interface unifiée pour tous les besoins Elasticsearch du service.

Classes principales :
- ElasticsearchQueryBuilder : Construction de requêtes optimisées
- ResponseFormatter : Formatage des réponses ES
- ErrorHandler : Gestion des erreurs ES
- IndexManager : Gestion des index
- QueryOptimizer : Optimisations avancées

Utilisé par :
- query_executor pour construire les requêtes
- lexical_engine pour le formatage des réponses
- clients/elasticsearch_client pour la gestion d'erreurs
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Imports locaux - à adapter selon la structure des models
try:
    from search_service.models.requests import InternalSearchRequest
    from search_service.models.filters import TermFilter, FilterOperator, AggregationType
    from search_service.models.elasticsearch_queries import TextQuery, FieldBoost
except ImportError:
    # Fallback si les models ne sont pas encore implémentés
    InternalSearchRequest = None
    TermFilter = None
    FilterOperator = None
    AggregationType = None
    TextQuery = None
    FieldBoost = None

try:
    from search_service.config import settings
    FIELD_CONFIGURATIONS = getattr(settings, 'FIELD_CONFIGURATIONS', {})
except ImportError:
    # Configuration par défaut
    FIELD_CONFIGURATIONS = {}
    class settings:
        max_pagination_offset = 10000


logger = logging.getLogger(__name__)


# === EXCEPTIONS SPÉCIALISÉES ===

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


# === ENUMS ===

class QueryOptimization(str, Enum):
    """Types d'optimisations appliquées aux requêtes"""
    USER_FILTER_FIRST = "user_filter_first"
    BOOST_OPTIMIZATION = "boost_optimization"
    FIELD_FILTERING = "field_filtering"
    QUERY_SIMPLIFICATION = "query_simplification"
    CACHE_FRIENDLY = "cache_friendly"
    AGGREGATION_OPTIMIZATION = "aggregation_optimization"


class QueryType(str, Enum):
    """Types de requêtes supportés"""
    SIMPLE_TERM = "simple_term"
    TEXT_SEARCH = "text_search"
    BOOLEAN_FILTER = "boolean_filter"
    AGGREGATION = "aggregation"
    COMPLEX_SEARCH = "complex_search"


class FieldType(str, Enum):
    """Types de champs Elasticsearch"""
    TEXT = "text"
    KEYWORD = "keyword"
    INTEGER = "integer"
    DOUBLE = "double"
    DATE = "date"
    BOOLEAN = "boolean"


# === STRUCTURES DE DONNÉES ===

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


@dataclass
class FieldConfiguration:
    """Configuration d'un champ pour les requêtes"""
    field_name: str
    field_type: FieldType
    is_searchable: bool = False
    is_filterable: bool = False
    supports_aggregation: bool = False
    boost: float = 1.0
    analyzer: Optional[str] = None


# === CONFIGURATION PAR DÉFAUT ===

DEFAULT_FIELD_CONFIGS = {
    "user_id": FieldConfiguration("user_id", FieldType.INTEGER, False, True, True, 1.0),
    "transaction_id": FieldConfiguration("transaction_id", FieldType.KEYWORD, False, True, True, 1.0),
    "amount": FieldConfiguration("amount", FieldType.DOUBLE, False, True, True, 1.0),
    "amount_abs": FieldConfiguration("amount_abs", FieldType.DOUBLE, False, True, True, 1.0),
    "date": FieldConfiguration("date", FieldType.DATE, False, True, True, 1.0),
    "searchable_text": FieldConfiguration("searchable_text", FieldType.TEXT, True, False, False, 2.0, "standard"),
    "primary_description": FieldConfiguration("primary_description", FieldType.TEXT, True, False, False, 1.5, "standard"),
    "merchant_name": FieldConfiguration("merchant_name", FieldType.TEXT, True, True, True, 1.8, "standard"),
    "category_name": FieldConfiguration("category_name", FieldType.TEXT, True, True, True, 1.0, "standard"),
    "transaction_type": FieldConfiguration("transaction_type", FieldType.KEYWORD, False, True, True, 1.0),
    "currency_code": FieldConfiguration("currency_code", FieldType.KEYWORD, False, True, True, 1.0),
    "operation_type": FieldConfiguration("operation_type", FieldType.KEYWORD, False, True, True, 1.0),
    "month_year": FieldConfiguration("month_year", FieldType.KEYWORD, False, True, True, 1.0),
    "weekday": FieldConfiguration("weekday", FieldType.KEYWORD, False, True, True, 1.0),
}


# === CLASSE PRINCIPALE - ElasticsearchQueryBuilder ===

class ElasticsearchQueryBuilder:
    """
    Constructeur de requêtes Elasticsearch optimisées
    Classe principale utilisée par query_executor
    """
    
    def __init__(self):
        self.optimizations_applied = []
        self.field_configs = {**DEFAULT_FIELD_CONFIGS, **FIELD_CONFIGURATIONS}
    
    def build_query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch à partir de données de requête
        Point d'entrée principal utilisé par query_executor
        
        Args:
            request_data: Données de requête (dict ou InternalSearchRequest)
            
        Returns:
            Dict: Requête Elasticsearch complète
        """
        self.optimizations_applied.clear()
        
        # Adapter selon le type d'entrée
        if hasattr(request_data, '__dict__'):
            # C'est un objet InternalSearchRequest
            return self.build_from_internal_request(request_data)
        else:
            # C'est un dictionnaire
            return self.build_from_dict(request_data)
    
    def build_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Construit une requête à partir d'un dictionnaire"""
        
        # Structure de base
        query_body = {
            "query": self._build_query_clause_from_dict(data),
            "size": data.get("limit", 20),
            "from": data.get("offset", 0),
            "_source": self._determine_source_fields_from_dict(data)
        }
        
        # Ajouter tri
        query_body["sort"] = self._build_sort_clause_from_dict(data)
        
        # Ajouter agrégations si demandées
        if data.get("aggregation_fields") or data.get("aggregation_types"):
            query_body["aggs"] = self._build_aggregations_from_dict(data)
            self.optimizations_applied.append(QueryOptimization.AGGREGATION_OPTIMIZATION)
        
        # Ajouter highlighting si demandé
        if data.get("include_highlights") and data.get("text_query"):
            query_body["highlight"] = self._build_highlight_config_from_dict(data)
        
        # Optimisations de performance
        query_body = self._apply_performance_optimizations(query_body, data)
        
        logger.debug(f"Built ES query with optimizations: {self.optimizations_applied}")
        return query_body
    
    def build_from_internal_request(self, request) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch à partir d'une InternalSearchRequest
        Conservé pour compatibilité future
        """
        if request is None:
            raise ValueError("InternalSearchRequest is None")
        
        # Pour l'instant, convertir en dict et utiliser build_from_dict
        request_dict = self._convert_request_to_dict(request)
        return self.build_from_dict(request_dict)
    
    def _convert_request_to_dict(self, request) -> Dict[str, Any]:
        """Convertit un InternalSearchRequest en dictionnaire"""
        try:
            if hasattr(request, '__dict__'):
                return request.__dict__
            else:
                return {}
        except Exception:
            return {}
    
    def _build_query_clause_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Construit la clause query principale depuis un dictionnaire"""
        bool_query = {
            "bool": {
                "must": [],
                "filter": [],
                "should": []
            }
        }
        
        # 1. Filtres de base
        filters = data.get("filters", [])
        user_id = data.get("user_id")
        
        # Assurer que user_id est toujours en premier
        if user_id:
            bool_query["bool"]["filter"].insert(0, {"term": {"user_id": user_id}})
            self.optimizations_applied.append(QueryOptimization.USER_FILTER_FIRST)
        
        # Ajouter autres filtres
        for filter_item in filters:
            es_filter = self._convert_filter_to_es(filter_item)
            if es_filter:
                bool_query["bool"]["filter"].append(es_filter)
        
        # 2. Recherche textuelle
        query_text = data.get("query_text") or data.get("text_query", {}).get("query", "")
        if query_text:
            text_queries = self._build_text_search_queries_from_dict(data)
            bool_query["bool"]["should"].extend(text_queries)
            
            if text_queries:
                bool_query["bool"]["minimum_should_match"] = 1
        
        # Optimisation: simplifier la structure si possible
        if not bool_query["bool"]["should"]:
            del bool_query["bool"]["should"]
            if "minimum_should_match" in bool_query["bool"]:
                del bool_query["bool"]["minimum_should_match"]
            self.optimizations_applied.append(QueryOptimization.QUERY_SIMPLIFICATION)
        
        if not bool_query["bool"]["must"]:
            del bool_query["bool"]["must"]
        
        return bool_query
    
    def _convert_filter_to_es(self, filter_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convertit un filtre en format Elasticsearch"""
        if not isinstance(filter_item, dict):
            return None
        
        field = filter_item.get("field")
        value = filter_item.get("value")
        operator = filter_item.get("operator", "eq")
        
        if not field or value is None:
            return None
        
        if operator == "eq":
            return {"term": {field: value}}
        elif operator == "in":
            return {"terms": {field: value if isinstance(value, list) else [value]}}
        elif operator == "gt":
            return {"range": {field: {"gt": value}}}
        elif operator == "gte":
            return {"range": {field: {"gte": value}}}
        elif operator == "lt":
            return {"range": {field: {"lt": value}}}
        elif operator == "lte":
            return {"range": {field: {"lte": value}}}
        elif operator == "between" and isinstance(value, list) and len(value) == 2:
            return {"range": {field: {"gte": value[0], "lte": value[1]}}}
        elif operator == "exists":
            if value:
                return {"exists": {"field": field}}
            else:
                return {"bool": {"must_not": {"exists": {"field": field}}}}
        elif operator == "prefix":
            return {"prefix": {field: value}}
        else:
            logger.warning(f"Unsupported filter operator: {operator}")
            return None
    
    def _build_text_search_queries_from_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construit les requêtes de recherche textuelle depuis un dictionnaire"""
        queries = []
        
        query_text = data.get("query_text") or data.get("text_query", {}).get("query", "")
        if not query_text:
            return queries
        
        # Champs de recherche par défaut
        search_fields = data.get("search_fields", [
            "searchable_text^2.0",
            "primary_description^1.5", 
            "merchant_name^1.8"
        ])
        
        # 1. Multi-match principal
        multi_match = {
            "multi_match": {
                "query": query_text,
                "fields": search_fields,
                "type": "best_fields",
                "operator": "or"
            }
        }
        
        fuzziness = data.get("fuzziness")
        if fuzziness:
            multi_match["multi_match"]["fuzziness"] = fuzziness
        
        queries.append(multi_match)
        
        # 2. Match phrase pour correspondances exactes
        phrase_query = {
            "multi_match": {
                "query": query_text,
                "fields": search_fields,
                "type": "phrase",
                "boost": 2.0
            }
        }
        queries.append(phrase_query)
        
        # 3. Match phrase prefix pour autocomplétion (requêtes courtes)
        if len(query_text.split()) <= 3:
            prefix_query = {
                "multi_match": {
                    "query": query_text,
                    "fields": search_fields,
                    "type": "phrase_prefix",
                    "boost": 1.5
                }
            }
            queries.append(prefix_query)
        
        self.optimizations_applied.append(QueryOptimization.BOOST_OPTIMIZATION)
        return queries
    
    def _determine_source_fields_from_dict(self, data: Dict[str, Any]) -> Union[bool, List[str]]:
        """Détermine les champs à retourner depuis un dictionnaire"""
        return_fields = data.get("return_fields")
        if return_fields:
            return return_fields
        
        # Champs par défaut
        default_fields = [
            "transaction_id", "user_id", "account_id",
            "amount", "amount_abs", "transaction_type", "currency_code",
            "date", "primary_description", "merchant_name", "category_name",
            "operation_type", "month_year", "weekday"
        ]
        
        self.optimizations_applied.append(QueryOptimization.FIELD_FILTERING)
        return default_fields
    
    def _build_sort_clause_from_dict(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construit la clause de tri depuis un dictionnaire"""
        sort_clause = []
        
        # Si recherche textuelle, trier par score d'abord
        if data.get("query_text") or data.get("text_query"):
            sort_clause.append({"_score": {"order": "desc"}})
        
        # Tri personnalisé
        custom_sort = data.get("sort")
        if custom_sort:
            if isinstance(custom_sort, list):
                sort_clause.extend(custom_sort)
            else:
                sort_clause.append(custom_sort)
        else:
            # Tri par défaut : date décroissante
            sort_clause.append({
                "date": {
                    "order": "desc",
                    "unmapped_type": "date",
                    "missing": "_last"
                }
            })
            
            # Tri de stabilité
            sort_clause.append({
                "transaction_id.keyword": {
                    "order": "desc",
                    "unmapped_type": "keyword"
                }
            })
        
        return sort_clause
    
    def _build_aggregations_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les agrégations depuis un dictionnaire"""
        aggs = {}
        
        # Agrégations par champ
        agg_fields = data.get("aggregation_fields", [])
        for field in agg_fields:
            if field in ["category_name", "merchant_name"]:
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
                        "size": 24,
                        "order": {"_key": "desc"}
                    },
                    "aggs": {
                        "total_amount": {"sum": {"field": "amount_abs"}},
                        "transaction_count": {"value_count": {"field": "transaction_id"}}
                    }
                }
        
        # Agrégations par type
        agg_types = data.get("aggregation_types", [])
        for agg_type in agg_types:
            if agg_type == "sum":
                aggs["total_sum"] = {"sum": {"field": "amount_abs"}}
            elif agg_type == "avg":
                aggs["average_amount"] = {"avg": {"field": "amount_abs"}}
            elif agg_type == "count":
                aggs["total_count"] = {"value_count": {"field": "transaction_id"}}
            elif agg_type == "stats":
                aggs["amount_stats"] = {"stats": {"field": "amount_abs"}}
            elif agg_type == "min":
                aggs["min_amount"] = {"min": {"field": "amount_abs"}}
            elif agg_type == "max":
                aggs["max_amount"] = {"max": {"field": "amount_abs"}}
        
        return aggs
    
    def _build_highlight_config_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Construit la configuration de highlighting depuis un dictionnaire"""
        highlight_fields = {}
        
        # Champs par défaut pour highlighting
        default_highlight_fields = ["searchable_text", "primary_description", "merchant_name"]
        
        for field in default_highlight_fields:
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
                    "number_of_fragments": 0
                }
        
        return {
            "fields": highlight_fields,
            "require_field_match": False
        }
    
    def _apply_performance_optimizations(self, query_body: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Applique les optimisations de performance"""
        
        # Cache de requête
        cache_strategy = data.get("cache_strategy", "standard")
        if cache_strategy in ["aggressive", "standard"]:
            query_body["request_cache"] = True
            self.optimizations_applied.append(QueryOptimization.CACHE_FRIENDLY)
        
        # Timeout adaptatif
        timeout_ms = data.get("timeout_ms")
        if timeout_ms:
            query_body["timeout"] = f"{timeout_ms}ms"
        
        # Track total hits optimisé
        limit = data.get("limit", 20)
        if limit <= 100:
            query_body["track_total_hits"] = True
        else:
            query_body["track_total_hits"] = 10000
        
        return query_body
    
    def get_applied_optimizations(self) -> List[str]:
        """Retourne la liste des optimisations appliquées"""
        return [opt.value for opt in self.optimizations_applied]
    
    def build_simple_user_query(self, user_id: int, limit: int = 20) -> Dict[str, Any]:
        """Construit une requête simple pour un utilisateur"""
        return self.build_from_dict({
            "user_id": user_id,
            "limit": limit,
            "filters": []
        })
    
    def build_text_search_query(self, user_id: int, query_text: str, 
                               fields: Optional[List[str]] = None,
                               limit: int = 20) -> Dict[str, Any]:
        """Construit une requête de recherche textuelle"""
        return self.build_from_dict({
            "user_id": user_id,
            "query_text": query_text,
            "search_fields": fields or ["searchable_text^2.0", "primary_description^1.5", "merchant_name^1.8"],
            "limit": limit,
            "include_highlights": True
        })


# === FORMATEUR DE RÉPONSES ===

class ResponseFormatter:
    """Formateur de réponses Elasticsearch"""
    
    @staticmethod
    def format_elasticsearch_response(es_response: Dict[str, Any]) -> ElasticsearchResponse:
        """
        Formate une réponse Elasticsearch brute en structure standardisée
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


# === GESTIONNAIRE D'ERREURS ===

class ErrorHandler:
    """Gestionnaire d'erreurs Elasticsearch spécialisé"""
    
    @staticmethod
    def parse_elasticsearch_error(error_response: Dict[str, Any], status_code: int) -> ElasticsearchError:
        """Parse une erreur Elasticsearch et la convertit en exception typée"""
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
        retryable_codes = [429, 502, 503, 504]
        retryable_types = ["timeout_exception", "es_rejected_execution_exception"]
        
        if error.status_code in retryable_codes:
            return True
        
        if error.es_error:
            error_type = error.es_error.get("error", {}).get("type", "")
            if error_type in retryable_types:
                return True
        
        return False


# === GESTIONNAIRE D'INDEX ===

class IndexManager:
    """Gestionnaire d'index Elasticsearch"""
    
    @staticmethod
    def get_optimal_index_settings() -> Dict[str, Any]:
        """Retourne les settings optimaux pour l'index de transactions"""
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "30s",
                "index": {
                    "max_result_window": getattr(settings, 'max_pagination_offset', 10000),
                    "mapping": {
                        "total_fields": {
                            "limit": 50
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


# === OPTIMISEUR DE REQUÊTES ===

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
            query_body["track_total_hits"] = True
        else:
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
    builder = ElasticsearchQueryBuilder()
    return builder.build_simple_user_query(user_id, limit)


def build_text_search_query(
    user_id: int, 
    query_text: str, 
    fields: Optional[List[str]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Construit une requête de recherche textuelle simple"""
    builder = ElasticsearchQueryBuilder()
    return builder.build_text_search_query(user_id, query_text, fields, limit)


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


# === ANALYSEURS SPÉCIALISÉS ===

class FieldAnalyzer:
    """Analyseur de champs pour optimisation des requêtes"""
    
    @staticmethod
    def get_searchable_fields() -> List[str]:
        """Retourne la liste des champs searchables"""
        searchable = []
        for field, config in DEFAULT_FIELD_CONFIGS.items():
            if config.is_searchable:
                searchable.append(field)
        return searchable
    
    @staticmethod
    def get_filterable_fields() -> List[str]:
        """Retourne la liste des champs filtrables"""
        filterable = []
        for field, config in DEFAULT_FIELD_CONFIGS.items():
            if config.is_filterable:
                filterable.append(field)
        return filterable
    
    @staticmethod
    def get_aggregatable_fields() -> List[str]:
        """Retourne la liste des champs agregables"""
        aggregatable = []
        for field, config in DEFAULT_FIELD_CONFIGS.items():
            if config.supports_aggregation:
                aggregatable.append(field)
        return aggregatable
    
    @staticmethod
    def suggest_field_optimization(field: str, operation: str) -> Optional[str]:
        """Suggère des optimisations pour un champ selon l'opération"""
        if field not in DEFAULT_FIELD_CONFIGS:
            return f"Field {field} not found in configuration"
        
        config = DEFAULT_FIELD_CONFIGS[field]
        
        if operation == "search" and not config.is_searchable:
            return f"Field {field} is not searchable. Consider using filterable fields."
        
        if operation == "filter" and not config.is_filterable:
            return f"Field {field} is not filterable."
        
        if operation == "aggregate" and not config.supports_aggregation:
            # Suggérer le champ .keyword si applicable
            if config.field_type == FieldType.TEXT:
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
    logger.debug(f"Elasticsearch query {context}:")
    logger.debug(json.dumps(query_body, indent=2, ensure_ascii=False))


# === EXPORTS ===

__all__ = [
    # === CLASSE PRINCIPALE ===
    "ElasticsearchQueryBuilder",  # AJOUTÉ - Classe principale utilisée par query_executor
    
    # === AUTRES CLASSES ===
    "ResponseFormatter", 
    "ErrorHandler",
    "IndexManager",
    "QueryOptimizer",
    "FieldAnalyzer",
    "QueryPerformanceAnalyzer",
    
    # === EXCEPTIONS ===
    "ElasticsearchError",
    
    # === STRUCTURES DE DONNÉES ===
    "ElasticsearchResponse",
    "FieldConfiguration",
    
    # === ENUMS ===
    "QueryOptimization",
    "QueryType",
    "FieldType",
    
    # === FONCTIONS UTILITAIRES ===
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
    
    # === DEBUGGING ===
    "explain_query_execution",
    "profile_query_execution", 
    "log_query_for_debug"
]