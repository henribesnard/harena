"""
Moteur de recherche lexicale pour Elasticsearch/Bonsai.

Ce module implémente la recherche lexicale complète avec requêtes optimisées,
filtres avancés, highlighting et évaluation de qualité.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from search_service.clients.elasticsearch_client import ElasticsearchClient
from search_service.core.query_processor import QueryProcessor, QueryAnalysis
from search_service.models.search_types import SearchType, SearchQuality, SortOrder
from search_service.models.responses import SearchResultItem
from search_service.utils.cache import SearchCache

logger = logging.getLogger(__name__)


@dataclass
class LexicalSearchConfig:
    """Configuration pour la recherche lexicale."""
    # Boost factors pour différents champs
    boost_exact_phrase: float = 10.0
    boost_merchant_name: float = 5.0
    boost_primary_description: float = 3.0
    boost_searchable_text: float = 4.0
    boost_clean_description: float = 2.5
    
    # Configuration des requêtes
    enable_fuzzy: bool = True
    enable_wildcards: bool = True
    enable_synonyms: bool = True
    minimum_should_match: str = "1"
    fuzziness_level: str = "AUTO"
    
    # Configuration du highlighting
    highlight_enabled: bool = True
    highlight_fragment_size: int = 150
    highlight_max_fragments: int = 3
    highlight_pre_tags: List[str] = None
    highlight_post_tags: List[str] = None
    
    # Filtres et seuils
    min_score_threshold: float = 1.0
    max_results: int = 50
    
    # Performance
    timeout_seconds: float = 5.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    
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
    """
    
    def __init__(
        self,
        elasticsearch_client: ElasticsearchClient,
        query_processor: Optional[QueryProcessor] = None,
        config: Optional[LexicalSearchConfig] = None
    ):
        self.elasticsearch_client = elasticsearch_client
        self.query_processor = query_processor or QueryProcessor()
        self.config = config or LexicalSearchConfig()
        
        # Cache pour les résultats
        self.cache = SearchCache(
            max_size=1000,
            ttl_seconds=self.config.cache_ttl_seconds
        ) if self.config.enable_cache else None
        
        # Métriques de performance
        self.search_count = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.failed_searches = 0
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        
        logger.info("Lexical search engine initialized")
    
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
            
            # 3. Exécuter la recherche avec timeout
            search_response = await asyncio.wait_for(
                self.elasticsearch_client.search(
                    index="harena_transactions",
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
        """Construit la requête Elasticsearch optimisée."""
        
        # Construire les clauses de requête avec boosting
        query_clauses = []
        
        # 1. Correspondance exacte de phrase (boost très élevé)
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
        
        # 2. Correspondance multi-champs principale
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
        
        # 3. CORRECTION: Recherche avec préfixes - utiliser les champs text (sans .keyword)
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
        
        # 4. Correspondances partielles avec wildcards
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
        
        # 6. Recherche fuzzy pour les variantes orthographiques
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
        
        # Ajouter le highlighting
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
            
            if range_filter["range"]["amount"]:  # Ajouter seulement si des limites sont définies
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
        
        # CORRECTION: Filtres par marchand - gestion sécurisée des champs keyword
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
            # Utiliser must_not pour exclure
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
                    if fragments  # Garder seulement les champs avec highlights
                }
            
            # Informations de debug
            explanation = None
            if debug and hit.get("_explanation"):
                explanation = {
                    "value": hit["_explanation"].get("value"),
                    "description": hit["_explanation"].get("description"),
                    "details": hit["_explanation"].get("details", [])[:3]  # Limiter les détails
                }
            
            # Construire les métadonnées
            metadata = {
                "search_engine": "lexical",
                "elasticsearch_score": hit["_score"],
                "index": hit["_index"],
                "doc_id": hit["_id"]
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
                lexical_score=hit["_score"],  # Pour recherche lexicale pure
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
            "max_score": search_response.get("hits", {}).get("max_score")
        }
    
    def _assess_lexical_quality(
        self,
        results: List[SearchResultItem],
        query_analysis: QueryAnalysis
    ) -> SearchQuality:
        """Évalue la qualité des résultats lexicaux."""
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
        
        # Conversion en enum de qualité
        if overall_quality >= 0.8:
            return SearchQuality.EXCELLENT
        elif overall_quality >= 0.6:
            return SearchQuality.GOOD
        elif overall_quality >= 0.4:
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
        
        # Score trop bas = mauvaise qualité
        if max_score <= self.config.min_score_threshold:
            return 0.2
        
        # Calculer la variance relative
        if len(scores) > 1:
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            relative_variance = variance / (avg_score ** 2) if avg_score > 0 else 0
        else:
            relative_variance = 0
        
        # Normaliser la qualité (score max entre 1-20 typiquement pour ES)
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
        
        # Bonus si les highlights sont riches (plusieurs champs)
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
            return 0.5  # Neutre si pas de résultats
        
        key_terms = getattr(query_analysis, 'key_terms', [])
        
        if not key_terms:
            return 0.5  # Neutre si pas de termes à analyser
        
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
            
            # Bonus pour la longueur du texte (plus d'informations)
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
        
        # Diversité par montant (répartition)
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
                index="harena_transactions",
                body=count_query
            )
            
            return response.get("count", 0)
        except Exception as e:
            logger.error(f"Failed to count documents for user {user_id}: {e}")
            return 0
    
    async def advanced_search(
        self,
        query: str,
        user_id: int,
        filters: Dict[str, Any],
        limit: int = 20,
        offset: int = 0
    ) -> LexicalSearchResult:
        """Effectue une recherche avancée avec filtres."""
        return await self.search(
            query=query,
            user_id=user_id,
            limit=limit,
            offset=offset,
            filters=filters,
            debug=False
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du moteur lexical."""
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
            "cache_stats": self.cache.get_stats() if self.cache else None
        }
    
    def clear_cache(self) -> None:
        """Vide le cache du moteur lexical."""
        if self.cache:
            self.cache.clear()
            logger.info("Lexical engine cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du moteur lexical."""
        try:
            # Test simple de connectivité
            health = await self.elasticsearch_client.health()
            
            return {
                "status": "healthy",
                "elasticsearch_status": health.get("status", "unknown"),
                "metrics": self.get_metrics()
            }
        except Exception as e:
            logger.error(f"Lexical engine health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "metrics": self.get_metrics()
            }
    
    async def get_suggestions(self, query: str, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Génère des suggestions basées sur une requête partielle."""
        try:
            # Construire une requête de suggestion sécurisée
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
                "size": 0  # Pas de documents, juste les agrégations
            }
            
            response = await self.elasticsearch_client.search(
                index="harena_transactions",
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
    
    async def analyze_query_performance(self, query: str, user_id: int) -> Dict[str, Any]:
        """Analyse les performances d'une requête spécifique."""
        try:
            start_time = time.time()
            
            # Exécuter la recherche avec profiling
            query_analysis = self.query_processor.process_query(query)
            optimized_query = self._optimize_query_for_elasticsearch(query_analysis)
            
            es_query = self._build_elasticsearch_query(
                optimized_query, query_analysis, user_id, None, True, SortOrder.RELEVANCE
            )
            
            # Ajouter le profiling
            es_query["profile"] = True
            
            search_response = await self.elasticsearch_client.search(
                index="harena_transactions",
                body=es_query,
                size=10
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Analyser les résultats du profiling
            profile_data = search_response.get("profile", {})
            shards = profile_data.get("shards", [])
            
            query_breakdown = []
            for shard in shards:
                searches = shard.get("searches", [])
                for search in searches:
                    query_info = search.get("query", [])
                    for q in query_info:
                        query_breakdown.append({
                            "type": q.get("type"),
                            "description": q.get("description"),
                            "time_in_nanos": q.get("time_in_nanos"),
                            "breakdown": q.get("breakdown", {})
                        })
            
            return {
                "original_query": query,
                "optimized_query": optimized_query,
                "execution_time_ms": execution_time,
                "elasticsearch_took_ms": search_response.get("took", 0),
                "total_hits": search_response.get("hits", {}).get("total", {}).get("value", 0),
                "max_score": search_response.get("hits", {}).get("max_score"),
                "query_breakdown": query_breakdown,
                "shard_count": len(shards),
                "query_analysis": {
                    "key_terms": getattr(query_analysis, 'key_terms', []),
                    "exact_phrases": getattr(query_analysis, 'exact_phrases', []),
                    "expanded_query": getattr(query_analysis, 'expanded_query', None)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze query performance: {e}")
            return {
                "error": str(e),
                "original_query": query
            }
    
    async def get_field_statistics(self, user_id: int) -> Dict[str, Any]:
        """Récupère les statistiques des champs pour un utilisateur."""
        try:
            stats_query = {
                "query": {
                    "term": {"user_id": user_id}
                },
                "aggs": {
                    "merchant_stats": {
                        "terms": {
                            "field": "merchant_name.keyword",
                            "size": 20
                        }
                    },
                    "category_stats": {
                        "terms": {
                            "field": "category_id",
                            "size": 20
                        }
                    },
                    "amount_stats": {
                        "stats": {
                            "field": "amount"
                        }
                    },
                    "transaction_type_stats": {
                        "terms": {
                            "field": "transaction_type",
                            "size": 10
                        }
                    },
                    "monthly_distribution": {
                        "date_histogram": {
                            "field": "transaction_date",
                            "calendar_interval": "month",
                            "format": "yyyy-MM"
                        }
                    }
                },
                "size": 0
            }
            
            response = await self.elasticsearch_client.search(
                index="harena_transactions",
                body=stats_query
            )
            
            aggregations = response.get("aggregations", {})
            
            return {
                "total_transactions": response.get("hits", {}).get("total", {}).get("value", 0),
                "top_merchants": [
                    {"name": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in aggregations.get("merchant_stats", {}).get("buckets", [])
                ],
                "categories": [
                    {"id": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in aggregations.get("category_stats", {}).get("buckets", [])
                ],
                "amount_statistics": aggregations.get("amount_stats", {}),
                "transaction_types": [
                    {"type": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in aggregations.get("transaction_type_stats", {}).get("buckets", [])
                ],
                "monthly_distribution": [
                    {"month": bucket["key_as_string"], "count": bucket["doc_count"]}
                    for bucket in aggregations.get("monthly_distribution", {}).get("buckets", [])
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get field statistics: {e}")
            return {"error": str(e)}
    
    def update_config(self, new_config: LexicalSearchConfig) -> None:
        """Met à jour la configuration du moteur."""
        old_config = self.config
        self.config = new_config
        
        # Recréer le cache si les paramètres ont changé
        if (old_config.cache_ttl_seconds != new_config.cache_ttl_seconds or
            old_config.enable_cache != new_config.enable_cache):
            
            if new_config.enable_cache:
                self.cache = SearchCache(
                    max_size=1000,
                    ttl_seconds=new_config.cache_ttl_seconds
                )
            else:
                self.cache = None
        
        logger.info("Lexical engine configuration updated")
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques de performance."""
        self.search_count = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.failed_searches = 0
        self.quality_distribution = {quality.value: 0 for quality in SearchQuality}
        
        logger.info("Lexical engine metrics reset")
    
    async def optimize_index(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Optimise l'index Elasticsearch pour de meilleures performances."""
        try:
            # Forcer un merge des segments
            await self.elasticsearch_client.indices.forcemerge(
                index="harena_transactions",
                max_num_segments=1
            )
            
            # Rafraîchir l'index
            await self.elasticsearch_client.indices.refresh(
                index="harena_transactions"
            )
            
            # Récupérer les statistiques post-optimisation
            stats = await self.elasticsearch_client.indices.stats(
                index="harena_transactions"
            )
            
            return {
                "status": "success",
                "index_stats": {
                    "total_docs": stats.get("_all", {}).get("total", {}).get("docs", {}).get("count", 0),
                    "total_size_bytes": stats.get("_all", {}).get("total", {}).get("store", {}).get("size_in_bytes", 0),
                    "segments_count": stats.get("_all", {}).get("total", {}).get("segments", {}).get("count", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def validate_mapping(self) -> Dict[str, Any]:
        """Valide le mapping de l'index Elasticsearch."""
        try:
            mapping = await self.elasticsearch_client.indices.get_mapping(
                index="harena_transactions"
            )
            
            properties = mapping.get("harena_transactions", {}).get("mappings", {}).get("properties", {})
            
            # Vérifier les champs critiques
            required_fields = {
                "user_id": "integer",
                "transaction_id": "keyword", 
                "primary_description": "text",
                "merchant_name": "text",
                "searchable_text": "text",
                "amount": "float",
                "transaction_date": "date"
            }
            
            validation_results = {}
            issues = []
            
            for field, expected_type in required_fields.items():
                if field not in properties:
                    issues.append(f"Missing required field: {field}")
                    validation_results[field] = {"status": "missing"}
                else:
                    field_config = properties[field]
                    actual_type = field_config.get("type")
                    
                    if actual_type == expected_type:
                        validation_results[field] = {"status": "ok", "type": actual_type}
                    else:
                        issues.append(f"Field {field} has type {actual_type}, expected {expected_type}")
                        validation_results[field] = {
                            "status": "type_mismatch",
                            "actual_type": actual_type,
                            "expected_type": expected_type
                        }
                    
                    # Vérifier la présence des champs .keyword pour les champs text
                    if expected_type == "text" and field_config.get("fields", {}).get("keyword"):
                        validation_results[field]["has_keyword_field"] = True
                    elif expected_type == "text":
                        issues.append(f"Field {field} missing .keyword subfield")
                        validation_results[field]["has_keyword_field"] = False
            
            return {
                "status": "valid" if not issues else "invalid",
                "issues": issues,
                "field_validation": validation_results,
                "total_fields": len(properties)
            }
            
        except Exception as e:
            logger.error(f"Failed to validate mapping: {e}")
            return {
                "status": "error",
                "error": str(e)
            }