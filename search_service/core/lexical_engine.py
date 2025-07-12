"""
Moteur de recherche lexicale pure Elasticsearch
Interface principale pour les recherches BM25 optimisées sur les données financières
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import time
import re

from models.service_contracts import SearchServiceQuery, SearchServiceResponse
from models.responses import InternalSearchResponse
from core.result_processor import result_processor_manager, ProcessingContext, ProcessingStrategy
from clients.elasticsearch_client import ElasticsearchClient
from utils.cache import LRUCache
from utils.metrics import LexicalSearchMetrics
from config import settings


logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Modes de recherche lexicale"""
    EXACT = "exact"
    FUZZY = "fuzzy"
    PHRASE = "phrase"
    BOOLEAN = "boolean"
    MULTI_MATCH = "multi_match"
    SEMANTIC_BOOST = "semantic_boost"


class FieldBoostStrategy(str, Enum):
    """Stratégies de boost des champs"""
    UNIFORM = "uniform"
    FINANCIAL = "financial"
    CONTEXTUAL = "contextual"
    DYNAMIC = "dynamic"


class QueryOptimization(str, Enum):
    """Types d'optimisations appliquées"""
    BM25_TUNING = "bm25_tuning"
    FIELD_BOOSTING = "field_boosting"
    PHRASE_DETECTION = "phrase_detection"
    SYNONYM_EXPANSION = "synonym_expansion"
    TYPO_TOLERANCE = "typo_tolerance"
    STOP_WORD_HANDLING = "stop_word_handling"
    USER_FILTER_OPTIMIZATION = "user_filter_optimization"
    QUERY_SIMPLIFICATION = "query_simplification"


@dataclass
class LexicalSearchConfig:
    """Configuration pour la recherche lexicale"""
    search_mode: SearchMode = SearchMode.MULTI_MATCH
    field_boost_strategy: FieldBoostStrategy = FieldBoostStrategy.FINANCIAL
    enable_fuzzy: bool = True
    fuzziness: str = "AUTO"
    minimum_should_match: str = "75%"
    tie_breaker: float = 0.3
    phrase_slop: int = 2
    max_expansions: int = 50
    prefix_length: int = 0
    boost_mode: str = "multiply"
    score_mode: str = "max"
    enable_highlighting: bool = True
    highlight_fragment_size: int = 150
    highlight_max_fragments: int = 3


@dataclass
class SearchContext:
    """Contexte d'exécution d'une recherche lexicale"""
    user_id: int
    query_text: Optional[str] = None
    search_intention: Optional[str] = None
    config: LexicalSearchConfig = field(default_factory=LexicalSearchConfig)
    performance_profile: str = "balanced"
    cache_enabled: bool = True
    debug_mode: bool = False
    request_id: Optional[str] = None


class FinancialFieldConfiguration:
    """Configuration des champs pour la recherche financière"""
    
    TEXT_FIELDS = {
        "searchable_text": {
            "boost": 2.0,
            "analyzer": "financial_analyzer",
            "description": "Texte enrichi principal"
        },
        "primary_description": {
            "boost": 1.5,
            "analyzer": "standard",
            "description": "Description transaction originale"
        },
        "merchant_name": {
            "boost": 1.8,
            "analyzer": "keyword_analyzer",
            "description": "Nom du marchand"
        },
        "category_name": {
            "boost": 1.2,
            "analyzer": "keyword_analyzer",
            "description": "Catégorie de transaction"
        }
    }
    
    FILTER_FIELDS = {
        "user_id": {"type": "term", "required": True},
        "category_name.keyword": {"type": "term", "boost": 1.1},
        "merchant_name.keyword": {"type": "term", "boost": 1.1},
        "transaction_type": {"type": "term"},
        "currency_code": {"type": "term"},
        "operation_type.keyword": {"type": "term"}
    }
    
    RANGE_FIELDS = {
        "amount": {"type": "float", "boost_large": True},
        "amount_abs": {"type": "float", "boost_large": True},
        "date": {"type": "date", "boost_recent": True}
    }
    
    FINANCIAL_SYNONYMS = {
        "restaurant": ["resto", "restauration", "café", "bar", "bistrot"],
        "supermarché": ["courses", "alimentation", "marché", "épicerie"],
        "carburant": ["essence", "diesel", "station", "sp95", "sp98"],
        "transport": ["métro", "bus", "train", "taxi", "uber"],
        "santé": ["pharmacie", "médecin", "hôpital", "clinique"],
        "banque": ["virement", "prélèvement", "commission", "frais"]
    }
    
    AMOUNT_PATTERNS = [
        r'\b\d+[.,]\d{2}\s*€?\b',
        r'\b\d+\s*euros?\b',
        r'\b\d+\s*€\b',
        r'\beuros?\s+\d+\b'
    ]


class LexicalQueryBuilder:
    """Constructeur de requêtes lexicales optimisées pour Elasticsearch"""
    
    def __init__(self, config: LexicalSearchConfig):
        self.config = config
        self.field_config = FinancialFieldConfiguration()
        self.optimizations_applied: List[QueryOptimization] = []
    
    def build_lexical_query(self, search_context: SearchContext,
                           filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Construit une requête lexicale optimisée"""
        
        query_body = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": [],
                    "should": [],
                    "must_not": []
                }
            },
            "sort": [],
            "highlight": {},
            "_source": self._get_source_fields(),
            "track_total_hits": True
        }
        
        # 1. Construire la partie textuelle
        if search_context.query_text:
            text_query = self._build_text_query(search_context.query_text, search_context)
            query_body["query"]["bool"]["must"].append(text_query)
        
        # 2. Ajouter les filtres obligatoires
        mandatory_filters = self._build_mandatory_filters(search_context.user_id)
        query_body["query"]["bool"]["filter"].extend(mandatory_filters)
        
        # 3. Ajouter les filtres optionnels
        if filters:
            optional_filters = self._build_optional_filters(filters)
            query_body["query"]["bool"]["filter"].extend(optional_filters)
        
        # 4. Configurer le tri
        sort_config = self._build_sort_configuration(search_context)
        query_body["sort"] = sort_config
        
        # 5. Configurer le highlighting
        if self.config.enable_highlighting:
            highlight_config = self._build_highlight_configuration(search_context)
            query_body["highlight"] = highlight_config
        
        # 6. Appliquer les optimisations de performance
        query_body = self._apply_performance_optimizations(query_body, search_context)
        
        # 7. Validation finale
        self._validate_query_structure(query_body)
        
        return query_body
    
    def _build_text_query(self, query_text: str, context: SearchContext) -> Dict[str, Any]:
        """Construit la partie recherche textuelle"""
        
        cleaned_query = self._preprocess_query_text(query_text)
        
        if self.config.search_mode == SearchMode.MULTI_MATCH:
            return self._build_multi_match_query(cleaned_query, context)
        elif self.config.search_mode == SearchMode.PHRASE:
            return self._build_phrase_query(cleaned_query, context)
        elif self.config.search_mode == SearchMode.FUZZY:
            return self._build_fuzzy_query(cleaned_query, context)
        elif self.config.search_mode == SearchMode.BOOLEAN:
            return self._build_boolean_query(cleaned_query, context)
        else:
            return self._build_multi_match_query(cleaned_query, context)
    
    def _build_multi_match_query(self, query_text: str, context: SearchContext) -> Dict[str, Any]:
        """Construit une requête multi_match optimisée"""
        
        fields_with_boosts = self._get_boosted_fields(context)
        
        base_query = {
            "multi_match": {
                "query": query_text,
                "fields": fields_with_boosts,
                "type": "best_fields",
                "tie_breaker": self.config.tie_breaker,
                "minimum_should_match": self.config.minimum_should_match
            }
        }
        
        if self.config.enable_fuzzy:
            base_query["multi_match"]["fuzziness"] = self.config.fuzziness
            base_query["multi_match"]["max_expansions"] = self.config.max_expansions
            base_query["multi_match"]["prefix_length"] = self.config.prefix_length
            self.optimizations_applied.append(QueryOptimization.TYPO_TOLERANCE)
        
        bool_query = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            **base_query["multi_match"],
                            "type": "phrase",
                            "boost": 3.0
                        }
                    },
                    {
                        "multi_match": {
                            **base_query["multi_match"],
                            "type": "best_fields",
                            "boost": 2.0
                        }
                    },
                    {
                        "multi_match": {
                            **base_query["multi_match"],
                            "type": "cross_fields",
                            "boost": 1.5,
                            "minimum_should_match": "50%"
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
        
        self.optimizations_applied.append(QueryOptimization.BM25_TUNING)
        return bool_query
    
    def _build_phrase_query(self, query_text: str, context: SearchContext) -> Dict[str, Any]:
        """Construit une requête de phrase avec slop"""
        
        fields_with_boosts = self._get_boosted_fields(context)
        
        phrase_query = {
            "multi_match": {
                "query": query_text,
                "fields": fields_with_boosts,
                "type": "phrase",
                "slop": self.config.phrase_slop
            }
        }
        
        self.optimizations_applied.append(QueryOptimization.PHRASE_DETECTION)
        return phrase_query
    
    def _build_fuzzy_query(self, query_text: str, context: SearchContext) -> Dict[str, Any]:
        """Construit une requête floue pour la tolérance aux typos"""
        
        terms = query_text.split()
        fuzzy_queries = []
        
        for term in terms:
            if len(term) > 3:
                for field, field_config in self.field_config.TEXT_FIELDS.items():
                    fuzzy_queries.append({
                        "fuzzy": {
                            field: {
                                "value": term,
                                "fuzziness": self.config.fuzziness,
                                "max_expansions": self.config.max_expansions,
                                "prefix_length": self.config.prefix_length,
                                "boost": field_config["boost"]
                            }
                        }
                    })
        
        if fuzzy_queries:
            self.optimizations_applied.append(QueryOptimization.TYPO_TOLERANCE)
            return {
                "bool": {
                    "should": fuzzy_queries,
                    "minimum_should_match": max(1, len(terms) // 2)
                }
            }
        else:
            return self._build_multi_match_query(query_text, context)
    
    def _build_boolean_query(self, query_text: str, context: SearchContext) -> Dict[str, Any]:
        """Construit une requête booléenne simple"""
        
        fields_with_boosts = self._get_boosted_fields(context)
        
        return {
            "simple_query_string": {
                "query": query_text,
                "fields": fields_with_boosts,
                "default_operator": "and",
                "analyze_wildcard": True,
                "minimum_should_match": self.config.minimum_should_match
            }
        }
    
    def _get_boosted_fields(self, context: SearchContext) -> List[str]:
        """Récupère les champs avec leurs boosts selon la stratégie"""
        
        if self.config.field_boost_strategy == FieldBoostStrategy.UNIFORM:
            return list(self.field_config.TEXT_FIELDS.keys())
        elif self.config.field_boost_strategy == FieldBoostStrategy.FINANCIAL:
            boosted_fields = []
            for field, config in self.field_config.TEXT_FIELDS.items():
                boost = config["boost"]
                boosted_fields.append(f"{field}^{boost}")
            self.optimizations_applied.append(QueryOptimization.FIELD_BOOSTING)
            return boosted_fields
        elif self.config.field_boost_strategy == FieldBoostStrategy.CONTEXTUAL:
            return self._get_contextual_boosted_fields(context)
        elif self.config.field_boost_strategy == FieldBoostStrategy.DYNAMIC:
            return self._get_dynamic_boosted_fields(context)
        else:
            return list(self.field_config.TEXT_FIELDS.keys())
    
    def _get_contextual_boosted_fields(self, context: SearchContext) -> List[str]:
        """Boosts contextuels selon l'intention de recherche"""
        
        base_boosts = {field: config["boost"] for field, config in self.field_config.TEXT_FIELDS.items()}
        
        if context.search_intention == "SEARCH_BY_MERCHANT":
            base_boosts["merchant_name"] *= 1.5
        elif context.search_intention == "SEARCH_BY_CATEGORY":
            base_boosts["category_name"] *= 1.5
            base_boosts["searchable_text"] *= 1.2
        elif context.search_intention == "TEXT_SEARCH":
            base_boosts["searchable_text"] *= 1.3
            base_boosts["primary_description"] *= 1.2
        
        return [f"{field}^{boost}" for field, boost in base_boosts.items()]
    
    def _get_dynamic_boosted_fields(self, context: SearchContext) -> List[str]:
        """Boosts dynamiques basés sur les performances historiques"""
        
        return self._get_boosted_fields(
            SearchContext(context.user_id, config=LexicalSearchConfig(
                field_boost_strategy=FieldBoostStrategy.FINANCIAL
            ))
        )
    
    def _preprocess_query_text(self, query_text: str) -> str:
        """Prétraite le texte de recherche"""
        
        cleaned = re.sub(r'[^\w\s€$£¥.,\-]', ' ', query_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        expanded = self._expand_synonyms(cleaned)
        
        if expanded != cleaned:
            self.optimizations_applied.append(QueryOptimization.SYNONYM_EXPANSION)
        
        return expanded
    
    def _expand_synonyms(self, query_text: str) -> str:
        """Expand les synonymes financiers"""
        
        expanded_terms = []
        terms = query_text.lower().split()
        
        for term in terms:
            expanded = [term]
            
            for main_term, synonyms in self.field_config.FINANCIAL_SYNONYMS.items():
                if term in synonyms:
                    expanded.append(main_term)
                elif term == main_term:
                    expanded.extend(synonyms[:2])
            
            if len(expanded) > 1:
                expanded_terms.append(f"({' OR '.join(expanded)})")
            else:
                expanded_terms.append(term)
        
        return ' '.join(expanded_terms)
    
    def _build_mandatory_filters(self, user_id: int) -> List[Dict[str, Any]]:
        """Construit les filtres obligatoires"""
        
        filters = [{"term": {"user_id": user_id}}]
        self.optimizations_applied.append(QueryOptimization.USER_FILTER_OPTIMIZATION)
        return filters
    
    def _build_optional_filters(self, filters_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Construit les filtres optionnels"""
        
        filters = []
        
        for field, value in filters_config.items():
            if field in self.field_config.FILTER_FIELDS:
                field_config = self.field_config.FILTER_FIELDS[field]
                
                if field_config["type"] == "term":
                    filter_query = {"term": {field: value}}
                    if "boost" in field_config:
                        filter_query = {
                            "constant_score": {
                                "filter": filter_query,
                                "boost": field_config["boost"]
                            }
                        }
                    filters.append(filter_query)
            elif field in self.field_config.RANGE_FIELDS:
                if isinstance(value, dict) and ("gte" in value or "lte" in value):
                    filters.append({"range": {field: value}})
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    filters.append({"range": {field: {"gte": value[0], "lte": value[1]}}})
        
        return filters
    
    def _build_sort_configuration(self, context: SearchContext) -> List[Dict[str, Any]]:
        """Configure le tri optimisé"""
        
        sort_config = []
        
        if context.query_text:
            sort_config.append({"_score": {"order": "desc"}})
        
        sort_config.append({
            "date": {
                "order": "desc",
                "unmapped_type": "date",
                "missing": "_last"
            }
        })
        
        sort_config.append({
            "transaction_id.keyword": {
                "order": "desc",
                "unmapped_type": "keyword"
            }
        })
        
        return sort_config
    
    def _build_highlight_configuration(self, context: SearchContext) -> Dict[str, Any]:
        """Configure le highlighting intelligent"""
        
        highlight_fields = {}
        
        for field, field_config in self.field_config.TEXT_FIELDS.items():
            if field in ["searchable_text", "primary_description"]:
                highlight_fields[field] = {
                    "fragment_size": self.config.highlight_fragment_size,
                    "number_of_fragments": self.config.highlight_max_fragments,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "fragmenter": "span"
                }
            elif field == "merchant_name":
                highlight_fields[field] = {
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "number_of_fragments": 0
                }
        
        return {
            "fields": highlight_fields,
            "require_field_match": False,
            "max_analyzed_offset": 1000000
        }
    
    def _apply_performance_optimizations(self, query_body: Dict[str, Any],
                                       context: SearchContext) -> Dict[str, Any]:
        """Applique les optimisations de performance"""
        
        if context.cache_enabled:
            query_body["request_cache"] = True
        
        if context.performance_profile == "fast":
            query_body["track_total_hits"] = 1000
            query_body["timeout"] = "100ms"
        elif context.performance_profile == "balanced":
            query_body["track_total_hits"] = True
            query_body["timeout"] = "500ms"
        else:
            query_body["track_total_hits"] = True
            query_body["timeout"] = "2s"
        
        query_body["preference"] = f"_local_{context.user_id}"
        
        self.optimizations_applied.append(QueryOptimization.QUERY_SIMPLIFICATION)
        return query_body
    
    def _get_source_fields(self) -> List[str]:
        """Définit les champs à retourner"""
        
        return [
            "transaction_id", "user_id", "account_id",
            "amount", "amount_abs", "transaction_type", "currency_code",
            "date", "primary_description", "merchant_name", "category_name",
            "operation_type", "month_year", "weekday", "searchable_text"
        ]
    
    def _validate_query_structure(self, query_body: Dict[str, Any]):
        """Valide la structure de la requête"""
        
        if not isinstance(query_body, dict):
            raise ValueError("Query body must be a dictionary")
        
        if "query" not in query_body:
            raise ValueError("Query body must contain a 'query' field")
        
        bool_query = query_body.get("query", {}).get("bool", {})
        filters = bool_query.get("filter", [])
        
        user_filter_found = any(
            "user_id" in str(filter_clause) for filter_clause in filters
        )
        
        if not user_filter_found:
            raise ValueError("user_id filter is required for security")


class LexicalSearchEngine:
    """Moteur de recherche lexicale principal"""
    
    def __init__(self, elasticsearch_client: ElasticsearchClient):
        self.es_client = elasticsearch_client
        self.metrics = LexicalSearchMetrics()
        self.cache = LRUCache(max_size=settings.LEXICAL_CACHE_SIZE)
        self._initialized = True
        
        logger.info("LexicalSearchEngine initialisé")
    
    async def search(self, search_request: SearchServiceQuery) -> SearchServiceResponse:
        """Interface principale de recherche lexicale"""
        
        start_time = time.time()
        request_id = search_request.query_metadata.query_id
        
        try:
            search_context = self._create_search_context(search_request)
            
            if search_context.cache_enabled:
                cache_key = self._generate_cache_key(search_request)
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    logger.debug(f"Cache hit pour la recherche {request_id}")
                    self.metrics.record_cache_hit(search_request.query_metadata.user_id)
                    return cached_response
            
            query_builder = LexicalQueryBuilder(search_context.config)
            es_query = query_builder.build_lexical_query(
                search_context,
                self._extract_filters_from_request(search_request)
            )
            
            es_response = await self.es_client.search(
                index=settings.ELASTICSEARCH_INDEX,
                body=es_query,
                size=search_request.search_parameters.limit,
                from_=search_request.search_parameters.offset
            )
            
            internal_response = self._convert_elasticsearch_response(
                es_response, search_request, query_builder.optimizations_applied
            )
            
            processing_context = ProcessingContext(
                user_id=search_request.query_metadata.user_id,
                query_text=self._extract_query_text(search_request),
                search_intention=search_request.query_metadata.intent_type,
                processing_strategy=ProcessingStrategy.ENHANCED
            )
            
            enhanced_response = result_processor_manager.process_complete_response(
                internal_response, processing_context
            )
            
            service_response = self._convert_to_service_response(
                enhanced_response, search_request, query_builder.optimizations_applied
            )
            
            if search_context.cache_enabled:
                self.cache.set(cache_key, service_response, ttl=settings.CACHE_TTL_SECONDS)
            
            execution_time = int((time.time() - start_time) * 1000)
            self.metrics.record_search_execution(
                user_id=search_request.query_metadata.user_id,
                query_type=search_request.search_parameters.query_type,
                execution_time_ms=execution_time,
                result_count=len(service_response.results),
                success=True
            )
            
            logger.info(f"Recherche lexicale complétée: {request_id}, "
                       f"{len(service_response.results)} résultats, {execution_time}ms")
            
            return service_response
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            self.metrics.record_search_execution(
                user_id=search_request.query_metadata.user_id,
                query_type=search_request.search_parameters.query_type,
                execution_time_ms=execution_time,
                result_count=0,
                success=False
            )
            
            logger.error(f"Erreur dans la recherche lexicale {request_id}: {str(e)}")
            return self._create_error_response(search_request, str(e))
    
    async def count(self, search_request: SearchServiceQuery) -> int:
        """Compte les résultats sans les récupérer"""
        
        try:
            search_context = self._create_search_context(search_request)
            query_builder = LexicalQueryBuilder(search_context.config)
            
            count_query = query_builder.build_lexical_query(
                search_context,
                self._extract_filters_from_request(search_request)
            )
            
            count_query.pop("sort", None)
            count_query.pop("highlight", None)
            count_query.pop("_source", None)
            
            count_response = await self.es_client.count(
                index=settings.ELASTICSEARCH_INDEX,
                body={"query": count_query["query"]}
            )
            
            return count_response.get("count", 0)
            
        except Exception as e:
            logger.error(f"Erreur lors du comptage: {str(e)}")
            return 0
    
    async def suggest(self, partial_query: str, user_id: int, limit: int = 5) -> List[str]:
        """Génère des suggestions de recherche"""
        
        try:
            suggestion_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}},
                            {"prefix": {"searchable_text": partial_query.lower()}}
                        ]
                    }
                },
                "aggs": {
                    "suggestions": {
                        "terms": {
                            "field": "searchable_text.keyword",
                            "size": limit * 2,
                            "include": f".*{re.escape(partial_query)}.*"
                        }
                    }
                },
                "size": 0
            }
            
            response = await self.es_client.search(
                index=settings.ELASTICSEARCH_INDEX,
                body=suggestion_query
            )
            
            suggestions = []
            if "aggregations" in response and "suggestions" in response["aggregations"]:
                for bucket in response["aggregations"]["suggestions"]["buckets"]:
                    suggestion = bucket["key"]
                    if len(suggestion) > len(partial_query) and len(suggestions) < limit:
                        suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de suggestions: {str(e)}")
            return []
    
    def _create_search_context(self, search_request: SearchServiceQuery) -> SearchContext:
        """Crée le contexte de recherche à partir du request"""
        
        config = LexicalSearchConfig()
        intent_type = search_request.query_metadata.intent_type
        
        if intent_type in ["TEXT_SEARCH", "TEXT_SEARCH_WITH_CATEGORY"]:
            config.search_mode = SearchMode.MULTI_MATCH
            config.field_boost_strategy = FieldBoostStrategy.CONTEXTUAL
        elif intent_type == "SEARCH_BY_MERCHANT":
            config.search_mode = SearchMode.PHRASE
            config.field_boost_strategy = FieldBoostStrategy.CONTEXTUAL
        elif intent_type == "SEARCH_BY_CATEGORY":
            config.search_mode = SearchMode.EXACT
            config.enable_fuzzy = False
        
        timeout_ms = search_request.search_parameters.timeout_ms
        if timeout_ms <= 100:
            performance_profile = "fast"
        elif timeout_ms <= 1000:
            performance_profile = "balanced"
        else:
            performance_profile = "comprehensive"
        
        return SearchContext(
            user_id=search_request.query_metadata.user_id,
            query_text=self._extract_query_text(search_request),
            search_intention=intent_type,
            config=config,
            performance_profile=performance_profile,
            cache_enabled=search_request.options.cache_enabled,
            debug_mode=search_request.options.include_explanation,
            request_id=search_request.query_metadata.query_id
        )
    
    def _extract_query_text(self, search_request: SearchServiceQuery) -> Optional[str]:
        """Extrait le texte de recherche du request"""
        
        if search_request.text_search:
            return search_request.text_search.query
        
        for filter_obj in search_request.filters.required:
            if filter_obj.field in ["searchable_text", "primary_description"]:
                return str(filter_obj.value)
        
        return None
    
    def _extract_filters_from_request(self, search_request: SearchServiceQuery) -> Dict[str, Any]:
        """Extrait les filtres du request"""
        
        filters = {}
        
        for filter_obj in search_request.filters.required:
            if filter_obj.field != "user_id":
                filters[filter_obj.field] = filter_obj.value
        
        for filter_obj in search_request.filters.optional:
            filters[filter_obj.field] = filter_obj.value
        
        for filter_obj in search_request.filters.ranges:
            if filter_obj.operator == "between" and isinstance(filter_obj.value, list):
                filters[filter_obj.field] = {
                    "gte": filter_obj.value[0],
                    "lte": filter_obj.value[1]
                }
            elif filter_obj.operator == "gte":
                filters[filter_obj.field] = {"gte": filter_obj.value}
            elif filter_obj.operator == "lte":
                filters[filter_obj.field] = {"lte": filter_obj.value}
            elif filter_obj.operator == "gt":
                filters[filter_obj.field] = {"gt": filter_obj.value}
            elif filter_obj.operator == "lt":
                filters[filter_obj.field] = {"lt": filter_obj.value}
        
        return filters
    
    def _convert_elasticsearch_response(self, es_response: Dict[str, Any],
                                      search_request: SearchServiceQuery,
                                      optimizations: List[QueryOptimization]) -> InternalSearchResponse:
        """Convertit la réponse Elasticsearch vers le format interne"""
        
        from models.responses import InternalSearchResponse, RawTransaction, ExecutionMetrics
        
        hits = es_response.get("hits", {}).get("hits", [])
        raw_results = []
        
        for hit in hits:
            source = hit["_source"]
            
            raw_transaction = RawTransaction(
                transaction_id=source.get("transaction_id", ""),
                user_id=source.get("user_id", 0),
                account_id=source.get("account_id", ""),
                amount=source.get("amount", 0.0),
                amount_abs=source.get("amount_abs", 0.0),
                transaction_type=source.get("transaction_type", ""),
                currency_code=source.get("currency_code", "EUR"),
                date=source.get("date", ""),
                primary_description=source.get("primary_description", ""),
                merchant_name=source.get("merchant_name", ""),
                category_name=source.get("category_name", ""),
                operation_type=source.get("operation_type", ""),
                month_year=source.get("month_year", ""),
                weekday=source.get("weekday", ""),
                searchable_text=source.get("searchable_text", ""),
                score=hit.get("_score", 0.0)
            )
            
            if "highlight" in hit:
                raw_transaction._highlights = hit["highlight"]
            
            raw_results.append(raw_transaction)
        
        total_hits = es_response.get("hits", {}).get("total", {})
        if isinstance(total_hits, dict):
            total_count = total_hits.get("value", 0)
        else:
            total_count = total_hits
        
        execution_metrics = ExecutionMetrics(
            total_time_ms=0,
            elasticsearch_took=es_response.get("took", 0),
            query_complexity="simple",
            optimizations_applied=[opt.value for opt in optimizations],
            cache_used=False,
            parallel_executions=1
        )
        
        internal_response = InternalSearchResponse(
            request_id=search_request.query_metadata.query_id,
            user_id=search_request.query_metadata.user_id,
            total_hits=total_count,
            returned_hits=len(raw_results),
            raw_results=raw_results,
            aggregations=[],
            execution_metrics=execution_metrics,
            quality_score=0.8,
            elasticsearch_response=es_response if search_request.options.include_explanation else None
        )
        
        return internal_response
    
    def _convert_to_service_response(self, internal_response: InternalSearchResponse,
                                   search_request: SearchServiceQuery,
                                   optimizations: List[QueryOptimization]) -> SearchServiceResponse:
        """Convertit vers le contrat de réponse de service"""
        
        from models.responses import ResponseTransformer
        
        service_response = ResponseTransformer.to_contract(internal_response)
        
        service_response.performance.optimization_applied.extend([opt.value for opt in optimizations])
        service_response.response_metadata.agent_context.update({
            "search_engine": "lexical",
            "elasticsearch_version": "8.x",
            "bm25_optimized": True
        })
        
        return service_response
    
    def _create_error_response(self, search_request: SearchServiceQuery, 
                              error_message: str) -> SearchServiceResponse:
        """Crée une réponse d'erreur"""
        
        from models.service_contracts import (
            SearchServiceResponse, ResponseMetadata, PerformanceMetrics, ContextEnrichment
        )
        
        return SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id=search_request.query_metadata.query_id,
                execution_time_ms=0,
                total_hits=0,
                returned_hits=0,
                has_more=False,
                cache_hit=False,
                elasticsearch_took=0,
                agent_context={
                    "error": error_message,
                    "search_engine": "lexical"
                }
            ),
            results=[],
            aggregations=None,
            performance=PerformanceMetrics(
                query_complexity="error",
                optimization_applied=["error_handling"],
                index_used=settings.ELASTICSEARCH_INDEX,
                shards_queried=0,
                cache_hit=False
            ),
            context_enrichment=ContextEnrichment(
                search_intent_matched=False,
                result_quality_score=0.0,
                suggested_followup_questions=["Vérifiez votre requête", "Essayez des termes différents"],
                next_suggested_agent="error_handler"
            )
        )
    
    def _generate_cache_key(self, search_request: SearchServiceQuery) -> str:
        """Génère une clé de cache pour la requête"""
        
        cache_elements = [
            f"user:{search_request.query_metadata.user_id}",
            f"intent:{search_request.query_metadata.intent_type}",
            f"limit:{search_request.search_parameters.limit}",
            f"offset:{search_request.search_parameters.offset}"
        ]
        
        query_text = self._extract_query_text(search_request)
        if query_text:
            cache_elements.append(f"text:{query_text}")
        
        filters = self._extract_filters_from_request(search_request)
        for field, value in sorted(filters.items()):
            cache_elements.append(f"{field}:{value}")
        
        return "lexical:" + "|".join(cache_elements)
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de recherche"""
        return {
            "lexical_engine": self.metrics.get_summary(),
            "cache_stats": self.cache.get_stats(),
            "elasticsearch_client": "connected" if self.es_client else "disconnected"
        }
    
    def clear_cache(self):
        """Vide le cache de recherche"""
        self.cache.clear()
        logger.info("Cache de recherche lexicale vidé")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du moteur lexical"""
        
        try:
            es_health = await self.es_client.cluster.health()
            
            test_query = {
                "query": {"match_all": {}},
                "size": 1
            }
            
            test_response = await self.es_client.search(
                index=settings.ELASTICSEARCH_INDEX,
                body=test_query
            )
            
            search_working = "hits" in test_response
            
            return {
                "status": "healthy" if search_working else "degraded",
                "elasticsearch": {
                    "status": es_health.get("status", "unknown"),
                    "cluster": es_health.get("cluster_name", "unknown")
                },
                "lexical_engine": {
                    "initialized": self._initialized,
                    "search_functional": search_working,
                    "cache_size": len(self.cache.cache)
                },
                "metrics": self.metrics.get_summary()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "elasticsearch": "unreachable"
            }


class LexicalEngineManager:
    """Gestionnaire global pour le moteur de recherche lexicale"""
    
    def __init__(self):
        self._elasticsearch_client = None
        self._lexical_engine = None
        self._initialized = False
    
    def initialize(self, elasticsearch_client: ElasticsearchClient):
        """Initialise le gestionnaire avec un client Elasticsearch"""
        self._elasticsearch_client = elasticsearch_client
        self._lexical_engine = LexicalSearchEngine(elasticsearch_client)
        self._initialized = True
        
        logger.info("LexicalEngineManager initialisé avec succès")
    
    @property
    def engine(self) -> LexicalSearchEngine:
        """Accès au moteur de recherche lexicale"""
        if not self._initialized:
            raise RuntimeError("LexicalEngineManager non initialisé. Appelez initialize() d'abord.")
        return self._lexical_engine
    
    async def search(self, search_request: SearchServiceQuery) -> SearchServiceResponse:
        """Interface de recherche principale"""
        return await self.engine.search(search_request)
    
    async def count(self, search_request: SearchServiceQuery) -> int:
        """Interface de comptage"""
        return await self.engine.count(search_request)
    
    async def suggest(self, partial_query: str, user_id: int, limit: int = 5) -> List[str]:
        """Interface de suggestions"""
        return await self.engine.suggest(partial_query, user_id, limit)
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérification de santé globale"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "error": "LexicalEngineManager not initialized"
            }
        
        return await self.engine.health_check()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance"""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        try:
            metrics = self.engine.get_search_metrics()
            
            return {
                "system_status": "active",
                "search_metrics": metrics,
                "configuration": {
                    "default_search_mode": SearchMode.MULTI_MATCH.value,
                    "default_boost_strategy": FieldBoostStrategy.FINANCIAL.value,
                    "cache_enabled": True,
                    "fuzzy_enabled": True
                },
                "field_configuration": {
                    "text_fields": list(FinancialFieldConfiguration.TEXT_FIELDS.keys()),
                    "filter_fields": list(FinancialFieldConfiguration.FILTER_FIELDS.keys()),
                    "range_fields": list(FinancialFieldConfiguration.RANGE_FIELDS.keys())
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def shutdown(self):
        """Arrêt propre du gestionnaire"""
        if self._initialized:
            try:
                self.engine.clear_cache()
                logger.info("LexicalEngineManager arrêté proprement")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt: {str(e)}")
            finally:
                self._initialized = False
                self._lexical_engine = None
                self._elasticsearch_client = None


# === INSTANCE GLOBALE ===
lexical_engine_manager = LexicalEngineManager()


# === FONCTIONS D'UTILITÉ PRINCIPALES ===

async def lexical_search(search_request: SearchServiceQuery) -> SearchServiceResponse:
    """Fonction principale pour la recherche lexicale"""
    return await lexical_engine_manager.search(search_request)


async def lexical_count(search_request: SearchServiceQuery) -> int:
    """Fonction principale pour le comptage lexical"""
    return await lexical_engine_manager.count(search_request)


async def lexical_suggest(partial_query: str, user_id: int, limit: int = 5) -> List[str]:
    """Fonction principale pour les suggestions lexicales"""
    return await lexical_engine_manager.suggest(partial_query, user_id, limit)


def initialize_lexical_engine(elasticsearch_client: ElasticsearchClient):
    """Initialise le système de recherche lexicale"""
    lexical_engine_manager.initialize(elasticsearch_client)


async def get_lexical_health() -> Dict[str, Any]:
    """Vérifie la santé du système de recherche lexicale"""
    return await lexical_engine_manager.health_check()


def shutdown_lexical_engine():
    """Arrêt propre du système de recherche lexicale"""
    lexical_engine_manager.shutdown()


# === FONCTIONS DE CONFIGURATION ===

def create_financial_search_config(
    search_mode: SearchMode = SearchMode.MULTI_MATCH,
    boost_strategy: FieldBoostStrategy = FieldBoostStrategy.FINANCIAL,
    enable_fuzzy: bool = True,
    performance_profile: str = "balanced"
) -> LexicalSearchConfig:
    """Crée une configuration de recherche financière optimisée"""
    
    config = LexicalSearchConfig(
        search_mode=search_mode,
        field_boost_strategy=boost_strategy,
        enable_fuzzy=enable_fuzzy
    )
    
    if performance_profile == "fast":
        config.minimum_should_match = "60%"
        config.max_expansions = 20
        config.enable_highlighting = False
    elif performance_profile == "comprehensive":
        config.minimum_should_match = "85%"
        config.max_expansions = 100
        config.highlight_max_fragments = 5
    
    return config


def get_optimized_field_boosts(intention: str) -> Dict[str, float]:
    """Retourne les boosts optimisés selon l'intention"""
    
    base_boosts = {field: config["boost"] for field, config in FinancialFieldConfiguration.TEXT_FIELDS.items()}
    
    if intention == "SEARCH_BY_MERCHANT":
        base_boosts["merchant_name"] *= 1.5
        base_boosts["searchable_text"] *= 1.2
    elif intention == "SEARCH_BY_CATEGORY":
        base_boosts["category_name"] *= 1.5
        base_boosts["searchable_text"] *= 1.3
    elif intention == "TEXT_SEARCH":
        base_boosts["searchable_text"] *= 1.4
        base_boosts["primary_description"] *= 1.2
    
    return base_boosts


def analyze_query_complexity(query_text: str) -> Dict[str, Any]:
    """Analyse la complexité d'une requête textuelle"""
    
    if not query_text:
        return {"complexity": "empty", "score": 0}
    
    terms = query_text.split()
    term_count = len(terms)
    
    has_amounts = bool(re.search(r'\d+[.,]\d{2}|\d+\s*€', query_text))
    has_dates = bool(re.search(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', query_text))
    has_operators = bool(re.search(r'\b(AND|OR|NOT)\b', query_text, re.IGNORECASE))
    
    complexity_score = term_count
    if has_amounts:
        complexity_score += 2
    if has_dates:
        complexity_score += 2
    if has_operators:
        complexity_score += 3
    
    if complexity_score <= 2:
        complexity = "simple"
    elif complexity_score <= 5:
        complexity = "moderate"
    elif complexity_score <= 10:
        complexity = "complex"
    else:
        complexity = "very_complex"
    
    return {
        "complexity": complexity,
        "score": complexity_score,
        "term_count": term_count,
        "has_amounts": has_amounts,
        "has_dates": has_dates,
        "has_operators": has_operators,
        "recommended_mode": SearchMode.BOOLEAN.value if has_operators else SearchMode.MULTI_MATCH.value
    }


def optimize_query_for_performance(query_text: str, performance_target: str) -> str:
    """Optimise une requête pour un objectif de performance"""
    
    if performance_target == "fast":
        terms = query_text.split()[:3]
        return " ".join(terms)
    elif performance_target == "comprehensive":
        config = FinancialFieldConfiguration()
        enriched_terms = []
        
        for term in query_text.split():
            enriched_terms.append(term)
            for main_term, synonyms in config.FINANCIAL_SYNONYMS.items():
                if term.lower() in synonyms:
                    enriched_terms.append(main_term)
                    break
        
        return " ".join(enriched_terms)
    else:
        return query_text


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # Classes principales
    "LexicalSearchEngine",
    "LexicalQueryBuilder",
    "LexicalEngineManager",
    "FinancialFieldConfiguration",
    
    # Enums
    "SearchMode",
    "FieldBoostStrategy",
    "QueryOptimization",
    
    # Modèles
    "LexicalSearchConfig",
    "SearchContext",
    
    # Fonctions principales
    "lexical_search",
    "lexical_count",
    "lexical_suggest",
    "initialize_lexical_engine",
    "get_lexical_health",
    "shutdown_lexical_engine",
    
    # Fonctions de configuration
    "create_financial_search_config",
    "get_optimized_field_boosts",
    "analyze_query_complexity",
    "optimize_query_for_performance",
    
    # Instance globale
    "lexical_engine_manager"
]


# === HELPERS ET UTILITAIRES ===

def get_lexical_components():
    """Retourne les composants du moteur lexical"""
    return {
        "manager": lexical_engine_manager,
        "engine": lexical_engine_manager.engine if lexical_engine_manager._initialized else None,
        "field_config": FinancialFieldConfiguration()
    }


def auto_initialize_lexical_engine():
    """Tentative d'initialisation automatique"""
    try:
        from clients.elasticsearch_client import get_default_client
        
        es_client = get_default_client()
        if es_client:
            initialize_lexical_engine(es_client)
            logger.info("LexicalEngine auto-initialisé avec succès")
            return True
    except ImportError:
        logger.debug("Client Elasticsearch non disponible pour auto-initialisation")
    except Exception as e:
        logger.warning(f"Échec auto-initialisation LexicalEngine: {str(e)}")
    
    return False


# === CONFIGURATION ET LOGGING ===

logger.info("LexicalEngine module chargé avec succès")