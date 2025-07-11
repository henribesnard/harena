"""
Moteur de recherche lexicale pour Elasticsearch/Bonsai - VERSION CENTRALISÉE.

Ce module implémente la recherche lexicale complète avec requêtes optimisées,
filtres avancés, highlighting et évaluation de qualité.

CENTRALISÉ VIA CONFIG_SERVICE:
- Toutes les configurations viennent de config_service.config.settings
- Boost factors, timeouts, seuils configurables
- Cache et highlighting selon la config centralisée
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ✅ CONFIGURATION CENTRALISÉE - SEULE SOURCE DE VÉRITÉ
from config_service.config import settings

from search_service.clients.elasticsearch_client import ElasticsearchClient
from search_service.core.query_processor import QueryProcessor, QueryAnalysis
from search_service.models.search_types import SearchType, SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem
from search_service.utils.cache import SearchCache

logger = logging.getLogger(__name__)


@dataclass
class LexicalSearchConfig:
    """Configuration pour la recherche lexicale - Basé sur config_service."""
    # Boost factors depuis config centralisée
    boost_exact_phrase: float = settings.BOOST_EXACT_PHRASE
    boost_merchant_name: float = settings.BOOST_MERCHANT_NAME
    boost_primary_description: float = settings.BOOST_PRIMARY_DESCRIPTION
    boost_searchable_text: float = settings.BOOST_SEARCHABLE_TEXT
    boost_clean_description: float = settings.BOOST_CLEAN_DESCRIPTION
    
    # Configuration des requêtes depuis config centralisée
    enable_fuzzy: bool = settings.ENABLE_FUZZY
    enable_wildcards: bool = settings.ENABLE_WILDCARDS
    enable_synonyms: bool = settings.ENABLE_SYNONYMS
    minimum_should_match: str = settings.MINIMUM_SHOULD_MATCH
    fuzziness_level: str = settings.FUZZINESS_LEVEL
    
    # Configuration du highlighting depuis config centralisée
    highlight_enabled: bool = settings.HIGHLIGHT_ENABLED
    highlight_fragment_size: int = settings.HIGHLIGHT_FRAGMENT_SIZE
    highlight_max_fragments: int = settings.HIGHLIGHT_MAX_FRAGMENTS
    highlight_pre_tags: List[str] = None
    highlight_post_tags: List[str] = None
    
    # Filtres et seuils depuis config centralisée
    min_score_threshold: float = settings.LEXICAL_MIN_SCORE
    max_results: int = settings.LEXICAL_MAX_RESULTS
    
    # Performance depuis config centralisée
    timeout_seconds: float = settings.ELASTICSEARCH_TIMEOUT
    enable_cache: bool = settings.SEARCH_CACHE_ENABLED
    cache_ttl_seconds: int = settings.SEARCH_CACHE_TTL
    
    def __post_init__(self):
        if self.highlight_pre_tags is None:
            self.highlight_pre_tags = ["<mark>"]
        if self.highlight_post_tags is None:
            self.highlight_post_tags = ["</mark>"]


@dataclass
class LexicalSearchResult:
    """Résultat d'une recherche lexicale."""
    results: List[SearchResultItem]
    total_found: int
    processing_time_ms: float
    query_used: str
    highlights_count: int
    quality: SearchQuality
    elasticsearch_query: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None


class LexicalSearchEngine:
    """
    Moteur de recherche lexicale utilisant Elasticsearch.
    
    Responsabilités:
    - Construction de requêtes Elasticsearch optimisées
    - Gestion des filtres et du highlighting
    - Correspondances exactes et floues
    - Boosting intelligent des champs
    - Évaluation de la qualité des résultats
    
    CONFIGURATION CENTRALISÉE VIA CONFIG_SERVICE.
    """
    
    def __init__(
        self,
        elasticsearch_client: ElasticsearchClient,
        query_processor: Optional[QueryProcessor] = None,
        config: Optional[LexicalSearchConfig] = None
    ):
        self.elasticsearch_client = elasticsearch_client
        self.query_processor = query_processor or QueryProcessor()
        
        # Configuration centralisée par défaut
        self.config = config or LexicalSearchConfig()
        
        # Cache pour les résultats avec config centralisée
        self.cache = SearchCache(
            max_size=settings.SEARCH_CACHE_SIZE,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_cache else None
        
        # Métriques de performance
        self.search_count = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.failed_searches = 0
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        
        logger.info("Lexical search engine initialized with centralized config")
    
    async def search(
        self,
        query: str,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        sort_order: SortOrder = SortOrder.RELEVANCE,
        filters: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> LexicalSearchResult:
        """
        Effectue une recherche lexicale dans Elasticsearch.
        
        Args:
            query: Terme de recherche
            user_id: ID de l'utilisateur
            limit: Nombre de résultats
            offset: Décalage pour pagination
            sort_order: Ordre de tri
            filters: Filtres additionnels
            debug: Inclure informations de debug
            
        Returns:
            Résultats de recherche lexicale
            
        Raises:
            Exception: Si la recherche échoue
        """
        start_time = time.time()
        self.search_count += 1
        
        # Limiter selon la config centralisée
        limit = min(limit, self.config.max_results)
        
        # Génération de la clé de cache
        cache_key = None
        if self.cache:
            cache_key = self._generate_cache_key(
                query, user_id, limit, offset, sort_order, filters
            )
            
            # Vérifier le cache
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                logger.debug(f"Cache hit for lexical search: {cache_key}")
                return cached_result
        
        try:
            # 1. Analyser et traiter la requête
            query_analysis = self.query_processor.process_query(query)
            optimized_query = self._optimize_query_for_elasticsearch(query_analysis)
            
            # 2. Construire la requête Elasticsearch
            es_query = self._build_elasticsearch_query(
                optimized_query, query_analysis, user_id, filters, debug, sort_order
            )
            
            # 3. Exécuter la recherche avec timeout configuré
            search_response = await asyncio.wait_for(
                self.elasticsearch_client.search(
                    index=settings.ELASTICSEARCH_INDEX,
                    body=es_query,
                    size=limit,
                    from_=offset
                ),
                timeout=self.config.timeout_seconds
            )
            
            # 4. Traiter les résultats
            results = self._process_elasticsearch_results(
                search_response, query_analysis, debug
            )
            
            # 5. Calculer les métriques
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            # 6. Évaluer la qualité
            quality = self._assess_lexical_quality(results, query_analysis)
            self.quality_distribution[quality.value] += 1
            
            # 7. Construire le résultat
            result = LexicalSearchResult(
                results=results,
                total_found=search_response.get("hits", {}).get("total", {}).get("value", 0),
                processing_time_ms=processing_time,
                query_used=optimized_query,
                highlights_count=sum(1 for r in results if r.highlights),
                quality=quality,
                elasticsearch_query=es_query if debug else None,
                debug_info=self._extract_debug_info(search_response) if debug else None
            )
            
            # 8. Mettre en cache si activé
            if self.cache and cache_key:
                self.cache.put(cache_key, result)
            
            return result
            
        except asyncio.TimeoutError:
            self.failed_searches += 1
            logger.error(f"Lexical search timeout after {self.config.timeout_seconds}s")
            raise Exception("Search timeout")
            
        except Exception as e:
            self.failed_searches += 1
            logger.error(f"Lexical search failed: {e}", exc_info=True)
            raise Exception(f"Lexical search error: {str(e)}")
    
    def _generate_cache_key(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        sort_order: SortOrder,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Génère une clé de cache pour la requête."""
        import hashlib
        
        # Créer une représentation unique de la requête
        cache_data = {
            "query": query.lower().strip(),
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
            "sort_order": sort_order.value,
            "filters": filters or {}
        }
        
        # Sérialiser et hasher
        cache_str = str(sorted(cache_data.items()))
        return f"lexical_{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def _optimize_query_for_elasticsearch(self, query_analysis: QueryAnalysis) -> str:
        """Optimise la requête pour Elasticsearch."""
        # Utiliser l'attribut expanded_query directement
        if hasattr(query_analysis, 'expanded_query') and query_analysis.expanded_query:
            return query_analysis.expanded_query
        
        # Sinon utiliser la requête nettoyée
        if hasattr(query_analysis, 'cleaned_query') and query_analysis.cleaned_query:
            return query_analysis.cleaned_query
            
        return query_analysis.original_query
    
    def _build_elasticsearch_query(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        user_id: int,
        filters: Optional[Dict[str, Any]],
        debug: bool,
        sort_order: SortOrder = SortOrder.RELEVANCE
    ) -> Dict[str, Any]:
        """Construit la requête Elasticsearch optimisée avec config centralisée."""
        
        # Construire les clauses de requête avec boosting configuré
        query_clauses = []
        
        # 1. Correspondance exacte de phrase (boost configuré)
        exact_phrases = getattr(query_analysis, 'exact_phrases', [])
        
        for phrase in exact_phrases:
            query_clauses.append({
                "multi_match": {
                    "query": phrase,
                    "fields": [
                        f"primary_description^{self.config.boost_exact_phrase}",
                        f"merchant_name^{self.config.boost_merchant_name * 1.5}",
                        f"searchable_text^{self.config.boost_searchable_text}"
                    ],
                    "type": "phrase",
                    "boost": self.config.boost_exact_phrase
                }
            })
        
        # 2. Correspondance multi-champs principale avec config centralisée
        query_clauses.append({
            "multi_match": {
                "query": query,
                "fields": [
                    f"primary_description^{self.config.boost_primary_description}",
                    f"merchant_name^{self.config.boost_merchant_name}",
                    f"searchable_text^{self.config.boost_searchable_text}",
                    f"clean_description^{self.config.boost_clean_description}"
                ],
                "type": "best_fields",
                "fuzziness": self.config.fuzziness_level if self.config.enable_fuzzy else "0",
                "minimum_should_match": self.config.minimum_should_match,
                "boost": 1.0
            }
        })
        
        # 3. Recherche avec préfixes si activée dans la config
        if self.config.enable_wildcards and len(query) > 2:
            query_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        f"primary_description^{self.config.boost_primary_description * 0.8}",
                        f"merchant_name^{self.config.boost_merchant_name * 0.8}",
                        "searchable_text^2.0"
                    ],
                    "type": "phrase_prefix",
                    "boost": 0.6
                }
            })
            
            # Ajouter une requête prefix séparée pour les champs keyword
            query_clauses.append({
                "bool": {
                    "should": [
                        {"prefix": {"merchant_name.keyword": {"value": query, "boost": 0.5}}},
                        {"prefix": {"primary_description.keyword": {"value": query, "boost": 0.4}}}
                    ]
                }
            })
        
        # 4. Correspondances partielles avec wildcards si activées
        if self.config.enable_wildcards and len(query) > 3:
            wildcard_query = f"*{query.lower()}*"
            query_clauses.append({
                "wildcard": {
                    "primary_description": {
                        "value": wildcard_query,
                        "boost": 0.4
                    }
                }
            })
            query_clauses.append({
                "wildcard": {
                    "merchant_name": {
                        "value": wildcard_query,
                        "boost": 0.5
                    }
                }
            })
        
        # 5. Recherche par termes individuels pour améliorer le rappel
        key_terms = getattr(query_analysis, 'key_terms', [])
        
        for term in key_terms:
            if len(term) > 2:  # Éviter les termes trop courts
                query_clauses.append({
                    "multi_match": {
                        "query": term,
                        "fields": [
                            "primary_description^1.5",
                            "merchant_name^2.0",
                            "searchable_text^1.2"
                        ],
                        "type": "cross_fields",
                        "boost": 0.3
                    }
                })
        
        # 6. Recherche fuzzy si activée dans la config
        if self.config.enable_fuzzy and len(query) > 4:
            query_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        "primary_description^1.8",
                        "merchant_name^2.2",
                        "searchable_text^1.5"
                    ],
                    "type": "most_fields",
                    "fuzziness": "AUTO",
                    "prefix_length": 1,
                    "boost": 0.7
                }
            })
        
        # Construction de la requête principale
        main_query = {
            "bool": {
                "should": query_clauses,
                "minimum_should_match": 1
            }
        }
        
        # Construire la requête complète avec filtres
        es_query = {
            "query": {
                "bool": {
                    "must": [main_query],
                    "filter": [
                        {"term": {"user_id": user_id}}
                    ]
                }
            },
            "min_score": self.config.min_score_threshold
        }
        
        # Ajouter les filtres additionnels
        if filters:
            self._add_filters_to_query(es_query, filters)
        
        # Ajouter le highlighting si activé dans la config
        if self.config.highlight_enabled:
            es_query["highlight"] = {
                "fields": {
                    "primary_description": {
                        "fragment_size": self.config.highlight_fragment_size,
                        "number_of_fragments": self.config.highlight_max_fragments,
                        "type": "unified"
                    },
                    "merchant_name": {
                        "fragment_size": 100,
                        "number_of_fragments": 1,
                        "type": "unified"
                    },
                    "searchable_text": {
                        "fragment_size": self.config.highlight_fragment_size,
                        "number_of_fragments": 2,
                        "type": "unified"
                    }
                },
                "pre_tags": self.config.highlight_pre_tags,
                "post_tags": self.config.highlight_post_tags,
                "require_field_match": False
            }
        
        # Ajouter le tri
        if sort_order != SortOrder.RELEVANCE:
            es_query["sort"] = self._build_sort_clause(sort_order)
        
        # Ajouter l'explication pour debug
        if debug:
            es_query["explain"] = True
        
        # ✅ FIX CRITIQUE: RETOURNER LA REQUÊTE CONSTRUITE
        return es_query
    
    def _add_filters_to_query(self, es_query: Dict[str, Any], filters: Dict[str, Any]) -> None:
        """Ajoute les filtres à la requête Elasticsearch."""
        filter_clauses = es_query["query"]["bool"]["filter"]
        
        # Filtres par type de transaction
        if filters.get("transaction_type"):
            filter_clauses.append({
                "term": {"transaction_type": filters["transaction_type"]}
            })
        
        # Filtres par montant
        if filters.get("amount_range"):
            amount_range = filters["amount_range"]
            range_filter = {"range": {"amount": {}}}
            
            if amount_range.get("min") is not None:
                range_filter["range"]["amount"]["gte"] = amount_range["min"]
            if amount_range.get("max") is not None:
                range_filter["range"]["amount"]["lte"] = amount_range["max"]
            
            if range_filter["range"]["amount"]:
                filter_clauses.append(range_filter)
        
        # Filtres par date
        if filters.get("date_range"):
            date_range = filters["date_range"]
            range_filter = {"range": {"transaction_date": {}}}
            
            if date_range.get("start"):
                range_filter["range"]["transaction_date"]["gte"] = date_range["start"]
            if date_range.get("end"):
                range_filter["range"]["transaction_date"]["lte"] = date_range["end"]
            
            if range_filter["range"]["transaction_date"]:
                filter_clauses.append(range_filter)
        
        # Filtres par compte
        if filters.get("account_ids") and filters["account_ids"]:
            filter_clauses.append({
                "terms": {"account_id": filters["account_ids"]}
            })
        
        # Filtres par catégorie
        if filters.get("category_ids") and filters["category_ids"]:
            filter_clauses.append({
                "terms": {"category_id": filters["category_ids"]}
            })
        
        # Filtres par marchand
        if filters.get("merchant_names") and filters["merchant_names"]:
            filter_clauses.append({
                "terms": {"merchant_name.keyword": filters["merchant_names"]}
            })
        
        # Filtre par correspondance partielle de marchand
        if filters.get("merchant_contains"):
            filter_clauses.append({
                "wildcard": {"merchant_name": f"*{filters['merchant_contains']}*"}
            })
        
        # Filtre par exclusion de marchands
        if filters.get("exclude_merchants") and filters["exclude_merchants"]:
            if "must_not" not in es_query["query"]["bool"]:
                es_query["query"]["bool"]["must_not"] = []
            
            es_query["query"]["bool"]["must_not"].append({
                "terms": {"merchant_name.keyword": filters["exclude_merchants"]}
            })
        
        # Filtre par période (derniers N jours)
        if filters.get("last_days"):
            filter_clauses.append({
                "range": {
                    "transaction_date": {
                        "gte": f"now-{filters['last_days']}d/d"
                    }
                }
            })
    
    def _build_sort_clause(self, sort_order: SortOrder) -> List[Dict[str, Any]]:
        """Construit la clause de tri."""
        if sort_order == SortOrder.DATE_DESC:
            return [
                {"transaction_date": {"order": "desc"}},
                {"_score": {"order": "desc"}}
            ]
        elif sort_order == SortOrder.DATE_ASC:
            return [
                {"transaction_date": {"order": "asc"}},
                {"_score": {"order": "desc"}}
            ]
        elif sort_order == SortOrder.AMOUNT_DESC:
            return [
                {"amount": {"order": "desc"}},
                {"_score": {"order": "desc"}}
            ]
        elif sort_order == SortOrder.AMOUNT_ASC:
            return [
                {"amount": {"order": "asc"}},
                {"_score": {"order": "desc"}}
            ]
        else:  # RELEVANCE par défaut
            return [{"_score": {"order": "desc"}}]
    
    def _process_elasticsearch_results(
        self,
        search_response: Dict[str, Any],
        query_analysis: QueryAnalysis,
        debug: bool
    ) -> List[SearchResultItem]:
        """Traite les résultats Elasticsearch en SearchResultItem."""
        results = []
        
        hits = search_response.get("hits", {}).get("hits", [])
        
        for hit in hits:
            source = hit["_source"]
            
            # Extraire les highlights
            highlights = None
            if hit.get("highlight"):
                highlights = {
                    field: fragments
                    for field, fragments in hit["highlight"].items()
                    if fragments
                }
            
            # Informations de debug
            explanation = None
            if debug and hit.get("_explanation"):
                explanation = {
                    "value": hit["_explanation"].get("value"),
                    "description": hit["_explanation"].get("description"),
                    "details": hit["_explanation"].get("details", [])[:3]
                }
            
            # Construire les métadonnées
            metadata = {
                "search_engine": "lexical",
                "elasticsearch_score": hit["_score"],
                "index": hit["_index"],
                "doc_id": hit["_id"],
                "config_source": "centralized (config_service)"
            }
            
            if debug:
                metadata["debug"] = {
                    "shard": hit.get("_shard"),
                    "node": hit.get("_node"),
                    "explanation": explanation
                }
            
            result_item = SearchResultItem(
                transaction_id=source["transaction_id"],
                user_id=source["user_id"],
                account_id=source.get("account_id"),
                score=hit["_score"],
                lexical_score=hit["_score"],
                semantic_score=None,
                combined_score=hit["_score"],
                primary_description=source["primary_description"],
                searchable_text=source.get("searchable_text"),
                merchant_name=source.get("merchant_name"),
                amount=source["amount"],
                currency_code=source.get("currency_code", "EUR"),
                transaction_type=source["transaction_type"],
                transaction_date=source["transaction_date"],
                created_at=source.get("created_at"),
                category_id=source.get("category_id"),
                operation_type=source.get("operation_type"),
                highlights=highlights,
                metadata=metadata,
                explanation=explanation
            )
            
            results.append(result_item)
        
        return results
    
    def _extract_debug_info(self, search_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait les informations de debug de la réponse Elasticsearch."""
        return {
            "took": search_response.get("took"),
            "timed_out": search_response.get("timed_out"),
            "shards": search_response.get("_shards"),
            "hits_total": search_response.get("hits", {}).get("total"),
            "max_score": search_response.get("hits", {}).get("max_score"),
            "config_source": "centralized (config_service)"
        }
    
    def _assess_lexical_quality(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité des résultats lexicaux avec seuils configurés."""
        if not results:
            return SearchQuality.POOR
        
        # Calculer différents aspects de qualité
        score_quality = self._assess_score_distribution(results)
        highlight_quality = self._assess_highlight_coverage(results)
        relevance_quality = self._assess_relevance_indicators(results, query_analysis)
        diversity_quality = self._assess_result_diversity(results)
        
        # Moyenne pondérée des qualités
        overall_quality = (
            score_quality * 0.35 +
            highlight_quality * 0.25 +
            relevance_quality * 0.30 +
            diversity_quality * 0.10
        )
        
        # Conversion en enum de qualité avec seuils configurés
        if overall_quality >= settings.QUALITY_EXCELLENT_THRESHOLD:
            return SearchQuality.EXCELLENT
        elif overall_quality >= settings.QUALITY_GOOD_THRESHOLD:
            return SearchQuality.GOOD
        elif overall_quality >= settings.QUALITY_MEDIUM_THRESHOLD:
            return SearchQuality.MEDIUM
        else:
            return SearchQuality.POOR
    
    def _assess_score_distribution(self, results: List[SearchResultItem]) -> float:
        """Évalue la distribution des scores."""
        if not results:
            return 0.0
        
        scores = [r.score for r in results if r.score]
        if not scores:
            return 0.0
        
        max_score = max(scores)
        min_score = min(scores)
        
        # Score trop bas = mauvaise qualité (seuil configuré)
        if max_score <= self.config.min_score_threshold:
            return 0.2
        
        # Calculer la variance relative
        if len(scores) > 1:
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            relative_variance = variance / (avg_score ** 2) if avg_score > 0 else 0
        else:
            relative_variance = 0
        
        # Normaliser la qualité
        normalized_max = min(max_score / 15.0, 1.0)
        
        # Bonus pour une bonne distribution des scores
        distribution_bonus = min(relative_variance * 2, 0.3)
        
        return min(normalized_max + distribution_bonus, 1.0)
    
    def _assess_highlight_coverage(self, results: List[SearchResultItem]) -> float:
        """Évalue la couverture des highlights."""
        if not results:
            return 0.0
        
        highlighted_results = sum(1 for r in results if r.highlights)
        coverage = highlighted_results / len(results)
        
        # Bonus si les highlights sont riches
        total_highlight_fields = sum(
            len(r.highlights) for r in results if r.highlights
        )
        
        if highlighted_results > 0:
            avg_fields_per_highlight = total_highlight_fields / highlighted_results
            field_bonus = min((avg_fields_per_highlight - 1) * 0.2, 0.3)
        else:
            field_bonus = 0
        
        return min(coverage + field_bonus, 1.0)
    
    def _assess_relevance_indicators(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> float:
        """Évalue les indicateurs de pertinence."""
        if not results:
            return 0.5
        
        key_terms = getattr(query_analysis, 'key_terms', [])
        
        if not key_terms:
            return 0.5
        
        relevance_scores = []
        
        for result in results:
            # Construire le texte à analyser
            text_to_check = " ".join(filter(None, [
                result.primary_description,
                result.merchant_name,
                result.searchable_text
            ])).lower()
            
            # Compter les termes qui matchent
            matching_terms = sum(
                1 for term in key_terms
                if term.lower() in text_to_check
            )
            
            # Calculer le ratio de pertinence
            term_coverage = matching_terms / len(key_terms) if key_terms else 0
            
            # Bonus pour la longueur du texte
            text_length_bonus = min(len(text_to_check) / 200, 0.2)
            
            result_relevance = min(term_coverage + text_length_bonus, 1.0)
            relevance_scores.append(result_relevance)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def _assess_result_diversity(self, results: List[SearchResultItem]) -> float:
        """Évalue la diversité des résultats."""
        if len(results) <= 1:
            return 1.0
        
        # Diversité par marchand
        merchants = {r.merchant_name for r in results if r.merchant_name}
        merchant_diversity = len(merchants) / len(results)
        
        # Diversité par catégorie
        categories = {r.category_id for r in results if r.category_id}
        category_diversity = len(categories) / len(results) if categories else 0.5
        
        # Diversité par montant
        amounts = [r.amount for r in results if r.amount]
        if len(amounts) > 1:
            amount_range = max(amounts) - min(amounts)
            amount_diversity = min(amount_range / max(amounts), 1.0) if max(amounts) > 0 else 0
        else:
            amount_diversity = 0.5
        
        # Moyenne pondérée
        return (merchant_diversity * 0.5 + category_diversity * 0.3 + amount_diversity * 0.2)
    
    async def count_user_documents(self, user_id: int) -> int:
        """Compte le nombre de documents pour un utilisateur."""
        try:
            count_query = {
                "query": {
                    "term": {"user_id": user_id}
                }
            }
            
            response = await self.elasticsearch_client.count(
                index=settings.ELASTICSEARCH_INDEX,
                body=count_query
            )
            
            return response.get("count", 0)
        except Exception as e:
            logger.error(f"Failed to count documents for user {user_id}: {e}")
            return 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moteur lexical avec info config centralisée."""
        avg_processing_time = (
            self.total_processing_time / self.search_count
            if self.search_count > 0 else 0
        )
        
        cache_hit_rate = self.cache_hits / self.search_count if self.search_count > 0 else 0
        failure_rate = self.failed_searches / self.search_count if self.search_count > 0 else 0
        
        return {
            "engine_type": "lexical",
            "search_count": self.search_count,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": avg_processing_time,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "failed_searches": self.failed_searches,
            "failure_rate": failure_rate,
            "quality_distribution": self.quality_distribution,
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "config_source": "centralized (config_service)",
            "centralized_settings": {
                "elasticsearch_timeout": settings.ELASTICSEARCH_TIMEOUT,
                "max_results": settings.LEXICAL_MAX_RESULTS,
                "min_score": settings.LEXICAL_MIN_SCORE,
                "cache_enabled": settings.SEARCH_CACHE_ENABLED,
                "cache_size": settings.SEARCH_CACHE_SIZE,
                "boost_factors": {
                    "exact_phrase": settings.BOOST_EXACT_PHRASE,
                    "merchant_name": settings.BOOST_MERCHANT_NAME,
                    "primary_description": settings.BOOST_PRIMARY_DESCRIPTION,
                    "searchable_text": settings.BOOST_SEARCHABLE_TEXT,
                    "clean_description": settings.BOOST_CLEAN_DESCRIPTION
                },
                "features": {
                    "fuzzy": settings.ENABLE_FUZZY,
                    "wildcards": settings.ENABLE_WILDCARDS,
                    "synonyms": settings.ENABLE_SYNONYMS,
                    "highlighting": settings.HIGHLIGHT_ENABLED
                }
            }
        }
    
    def clear_cache(self) -> None:
        """Vide le cache du moteur lexical."""
        if self.cache:
            self.cache.clear()
            logger.info("Lexical engine cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moteur lexical avec info config centralisée."""
        try:
            # Test simple de connectivité
            health = await self.elasticsearch_client.health()
            
            return {
                "status": "healthy",
                "elasticsearch_status": health.get("status", "unknown"),
                "elasticsearch_index": settings.ELASTICSEARCH_INDEX,
                "metrics": self.get_metrics(),
                "config_source": "centralized (config_service)"
            }
        except Exception as e:
            logger.error(f"Lexical engine health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.get_metrics(),
                "config_source": "centralized (config_service)"
            }
    
    def update_config(self, new_config: LexicalSearchConfig) -> None:
        """Met à jour la configuration du moteur."""
        old_config = self.config
        self.config = new_config
        
        # Recréer le cache si les paramètres ont changé
        if (old_config.cache_ttl_seconds != new_config.cache_ttl_seconds or
            old_config.enable_cache != new_config.enable_cache):
            
            if new_config.enable_cache:
                self.cache = SearchCache(
                    max_size=settings.SEARCH_CACHE_SIZE,
                    ttl_seconds=new_config.cache_ttl_seconds
                )
            else:
                self.cache = None
        
        logger.info("Lexical engine configuration updated (centralized config)")
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques de performance."""
        self.search_count = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.failed_searches = 0
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        
        logger.info("Lexical engine metrics reset")
    
    # Méthodes supplémentaires avec config centralisée...
    async def get_suggestions(self, query: str, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Génère des suggestions basées sur une requête partielle."""
        try:
            suggest_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"user_id": user_id}},
                            {
                                "bool": {
                                    "should": [
                                        {"prefix": {"primary_description": query.lower()}},
                                        {"prefix": {"merchant_name": query.lower()}},
                                        {"wildcard": {"primary_description": f"*{query.lower()}*"}},
                                        {"wildcard": {"merchant_name": f"*{query.lower()}*"}}
                                    ]
                                }
                            }
                        ]
                    }
                },
                "aggs": {
                    "description_suggestions": {
                        "terms": {
                            "field": "primary_description.keyword",
                            "size": limit,
                            "include": f".*{query.lower()}.*"
                        }
                    },
                    "merchant_suggestions": {
                        "terms": {
                            "field": "merchant_name.keyword", 
                            "size": limit,
                            "include": f".*{query.lower()}.*"
                        }
                    }
                },
                "size": 0
            }
            
            response = await self.elasticsearch_client.search(
                index=settings.ELASTICSEARCH_INDEX,
                body=suggest_query
            )
            
            suggestions = []
            
            # Suggestions de descriptions
            desc_buckets = response.get("aggregations", {}).get("description_suggestions", {}).get("buckets", [])
            for bucket in desc_buckets:
                suggestions.append({
                    "text": bucket["key"],
                    "frequency": bucket["doc_count"],
                    "type": "description"
                })
            
            # Suggestions de marchands
            merchant_buckets = response.get("aggregations", {}).get("merchant_suggestions", {}).get("buckets", [])
            for bucket in merchant_buckets:
                suggestions.append({
                    "text": bucket["key"],
                    "frequency": bucket["doc_count"],
                    "type": "merchant"
                })
            
            # Trier par fréquence décroissante et limiter
            suggestions.sort(key=lambda x: x["frequency"], reverse=True)
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get suggestions: {e}")
            return []