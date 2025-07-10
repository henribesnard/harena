"""
Client Elasticsearch/Bonsai pour le service de recherche.

Ce module fournit une interface optimisée pour interagir avec
Elasticsearch/Bonsai pour les recherches lexicales de transactions financières.

ARCHITECTURE:
- Hérite de BaseClient pour robustesse (retry, circuit breaker, health checks)
- Validation défensive stricte pour éviter les erreurs runtime
- Méthodes génériques compatibles avec lexical_engine.py
- Méthodes spécialisées pour les transactions financières
- Optimisations performance et cache intégré

RESPONSABILITÉS:
✅ Recherches lexicales optimisées domaine financier
✅ Construction de requêtes complexes avec validation
✅ Gestion des filtres et agrégations
✅ Highlighting des résultats
✅ Monitoring des performances et métriques
✅ Compatibilité avec l'architecture existante

USAGE:
    client = ElasticsearchClient(
        bonsai_url=settings.BONSAI_URL,
        index_name="harena_transactions"
    )
    await client.start()
    
    # Recherche générique (utilisée par lexical_engine.py)
    results = await client.search(
        index="harena_transactions",
        body=query_body,
        size=20,
        from_=0
    )
    
    # Recherche spécialisée transactions
    results = await client.search_transactions(
        query="virement café",
        user_id=123,
        limit=20,
        filters={"amount_min": 10.0}
    )
"""

import logging
import ssl
import time
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal

import aiohttp

from search_service.clients.base_client import (
    BaseClient, RetryConfig, CircuitBreakerConfig, HealthCheckConfig,
    ClientError, ConnectionError, TimeoutError
)
from search_service.utils.validators import QueryValidator, FilterValidator, ValidationError
from search_service.utils.cache import LRUCache, CacheKey
from search_service.utils.metrics import MetricsCollector, QueryTimer
from search_service.utils.elasticsearch_helpers import (
    ElasticsearchHelpers, QueryBuilder, ResultFormatter, 
    QueryStrategy, SortStrategy
)

# Configuration centralisée
from config_service.config import settings

logger = logging.getLogger(__name__)

# ==================== CONSTANTES ET CONFIGURATION ====================

# Synonymes financiers pour expansion de requêtes
FINANCIAL_SYNONYMS = {
    "virement": ["transfer", "transfert", "wire", "transfer bancaire", "vir"],
    "carte": ["card", "cb", "credit card", "debit card", "visa", "mastercard", "carte bancaire"],
    "retrait": ["withdrawal", "cash", "atm", "distributeur", "retrait especes"],
    "depot": ["deposit", "dépôt", "versement", "depot especes"],
    "prelevement": ["direct debit", "prélèvement automatique", "debit", "prelev"],
    "cheque": ["check", "chèque", "cheque bancaire"],
    "cafe": ["coffee", "café", "cafeteria", "cafétéria", "starbucks", "costa"],
    "restaurant": ["resto", "food", "meal", "dining", "restauration", "brasserie"],
    "essence": ["gas", "fuel", "station", "petrol", "shell", "total", "bp", "carburant"],
    "pharmacie": ["pharmacy", "drug store", "medication", "medicament", "parapharmacie"],
    "supermarche": ["supermarket", "grocery", "courses", "carrefour", "leclerc", "auchan"],
    "transport": ["metro", "bus", "train", "taxi", "uber", "sncf", "ratp", "transport public"]
}

# Champs optimisés pour la recherche financière
FINANCIAL_SEARCH_FIELDS = [
    "searchable_text^4.0",
    "primary_description^3.0",
    "clean_description^2.5",
    "provider_description^2.0",
    "merchant_name^3.5"
]

# Champs pour le highlighting
HIGHLIGHT_FIELDS = [
    "searchable_text",
    "primary_description",
    "merchant_name"
]

# Boost de scoring optimisés
BOOST_VALUES = {
    "exact_phrase": 10.0,
    "merchant_name": 8.0,
    "fuzzy_match": 4.0,
    "wildcard": 2.0,
    "synonym": 2.5,
    "simple_query": 1.5
}

# ==================== CLIENT ELASTICSEARCH ====================

class ElasticsearchClient(BaseClient):
    """
    Client pour Elasticsearch/Bonsai optimisé pour les transactions financières.
    
    Responsabilités:
    - Recherches lexicales optimisées pour le domaine financier
    - Construction de requêtes complexes avec validation stricte
    - Gestion des filtres et agrégations
    - Highlighting des résultats avec optimisation
    - Monitoring des performances et métriques
    - Compatibilité avec lexical_engine.py via méthodes génériques
    """
    
    def __init__(
        self,
        bonsai_url: str,
        index_name: str = "harena_transactions",
        timeout: float = 5.0,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        enable_cache: bool = True,
        enable_metrics: bool = True
    ):
        # Configuration SSL pour Bonsai
        health_check_config = HealthCheckConfig(
            enabled=True,
            interval_seconds=30.0,
            timeout_seconds=3.0,
            endpoint="/"
        )
        
        # Configuration par défaut optimisée pour Elasticsearch
        if retry_config is None:
            retry_config = RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=10.0,
                backoff_factor=2.0,
                jitter=True
            )
        
        if circuit_breaker_config is None:
            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=5,
                timeout_threshold=10.0,
                recovery_timeout=60.0
            )
        
        super().__init__(
            base_url=bonsai_url,
            service_name="elasticsearch",
            timeout=timeout,
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            health_check_config=health_check_config
        )
        
        self.index_name = index_name
        self.ssl_context = ssl.create_default_context()
        
        # Composants optionnels
        self.enable_cache = enable_cache
        self.enable_metrics = enable_metrics
        
        # Cache des requêtes fréquentes
        self._query_cache: Optional[LRUCache] = None
        if self.enable_cache:
            cache_size = getattr(settings, 'SEARCH_CACHE_SIZE', 1000)
            cache_ttl = getattr(settings, 'SEARCH_CACHE_TTL', 300)
            self._query_cache = LRUCache(
                max_size=cache_size,
                ttl_seconds=cache_ttl,
                name=f"elasticsearch_cache_{index_name}"
            )
        
        # Collecteur de métriques
        self._metrics_collector: Optional[MetricsCollector] = None
        if self.enable_metrics:
            self._metrics_collector = MetricsCollector(
                service_name="elasticsearch_client",
                include_query_metrics=True,
                include_performance_metrics=True,
                include_cache_metrics=True
            )
        
        # Validateurs
        self._query_validator = QueryValidator(
            max_query_length=getattr(settings, 'SEARCH_MAX_QUERY_LENGTH', 500),
            max_results_limit=getattr(settings, 'SEARCH_MAX_LIMIT', 100),
            allowed_fields=set(field.split('^')[0] for field in FINANCIAL_SEARCH_FIELDS)
        )
        self._filter_validator = FilterValidator()
        
        # Helpers
        self._query_builder = QueryBuilder(
            default_fields=FINANCIAL_SEARCH_FIELDS,
            highlight_fields=HIGHLIGHT_FIELDS,
            boost_merchant_name=BOOST_VALUES["merchant_name"],
            boost_exact_phrase=BOOST_VALUES["exact_phrase"]
        )
        self._result_formatter = ResultFormatter()
        
        logger.info(f"Elasticsearch client initialized for index: {index_name}")
    
    async def start(self):
        """Démarre le client avec configuration SSL pour Bonsai."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                ssl=self.ssl_context,
                limit=20,  # Pool de connexions
                limit_per_host=10
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self.headers
            )
            logger.info(f"{self.service_name} client started with SSL")
        
        # Démarrer les composants optionnels
        if self._query_cache:
            # Le cache démarre automatiquement
            pass
        
        # Vérifier la connectivité
        try:
            is_healthy = await self.test_connection()
            if not is_healthy:
                logger.warning("Elasticsearch connection test failed but continuing...")
        except Exception as e:
            logger.error(f"Elasticsearch startup error: {e}")
            # Ne pas lever d'exception pour permettre le démarrage en mode dégradé
    
    async def stop(self):
        """Arrête le client et nettoie les ressources."""
        # Arrêter le cache
        if self._query_cache:
            await self._query_cache.clear()
        
        # Arrêter le collecteur de métriques
        if self._metrics_collector:
            await self._metrics_collector.shutdown()
        
        # Arrêter le client de base
        await super().stop()
    
    async def test_connection(self) -> bool:
        """Teste la connectivité de base à Elasticsearch."""
        try:
            async def _test():
                async with self.session.get(self.base_url) as response:
                    if response.status == 200:
                        cluster_info = await response.json()
                        logger.info(f"✅ Connected to Elasticsearch cluster: {cluster_info.get('cluster_name', 'unknown')}")
                        return True
                    return False
            
            return await self.execute_with_retry(_test, "connection_test")
        except Exception as e:
            logger.error(f"Elasticsearch connection test failed: {e}")
            return False
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Effectue une vérification de santé spécifique à Elasticsearch."""
        try:
            # Vérifier le cluster
            async with self.session.get(self.base_url) as response:
                if response.status == 200:
                    cluster_info = await response.json()
                    
                    # Vérifier l'existence de l'index
                    index_exists = await self._check_index_exists()
                    
                    # Statistiques de l'index si disponible
                    index_stats = {}
                    if index_exists:
                        try:
                            index_stats = await self._get_basic_index_stats()
                        except Exception as e:
                            logger.debug(f"Could not get index stats: {e}")
                    
                    return {
                        "cluster_name": cluster_info.get("cluster_name", "unknown"),
                        "version": cluster_info.get("version", {}).get("number", "unknown"),
                        "index_exists": index_exists,
                        "index_stats": index_stats,
                        "status": "healthy" if index_exists else "degraded"
                    }
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def _check_index_exists(self) -> bool:
        """Vérifie si l'index existe."""
        try:
            async with self.session.head(f"{self.base_url}/{self.index_name}") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _get_basic_index_stats(self) -> Dict[str, Any]:
        """Récupère des statistiques de base de l'index."""
        try:
            async with self.session.get(f"{self.base_url}/{self.index_name}/_stats") as response:
                if response.status == 200:
                    stats = await response.json()
                    indices_stats = stats.get("indices", {}).get(self.index_name, {})
                    primaries = indices_stats.get("primaries", {})
                    
                    return {
                        "document_count": primaries.get("docs", {}).get("count", 0),
                        "store_size_bytes": primaries.get("store", {}).get("size_in_bytes", 0),
                        "indexing_total": primaries.get("indexing", {}).get("index_total", 0),
                        "search_total": primaries.get("search", {}).get("query_total", 0)
                    }
        except Exception:
            pass
        return {}
    
    # ============================================================================
    # MÉTHODES GÉNÉRIQUES COMPATIBLES AVEC LEXICAL_ENGINE.PY
    # ============================================================================
    
    async def search(
        self,
        index: str,
        body: Optional[Dict[str, Any]],
        size: int = 20,
        from_: int = 0
    ) -> Dict[str, Any]:
        """
        Effectue une recherche Elasticsearch (méthode générique).
        
        Cette méthode est utilisée par le lexical_engine.py et doit être
        compatible avec l'interface attendue.
        
        Args:
            index: Nom de l'index
            body: Corps de la requête Elasticsearch
            size: Nombre de résultats
            from_: Offset pour pagination
            
        Returns:
            Résultats de la recherche
            
        Raises:
            ValueError: Si body est None ou invalide
            ClientError: Si la recherche échoue
        """
        # ✅ VALIDATION DÉFENSIVE CRITIQUE
        if body is None:
            logger.error("Search body is None - this indicates a bug in query construction")
            logger.error(f"Called with index={index}, size={size}, from_={from_}")
            raise ValueError("Search body cannot be None. Check query construction in calling code.")
        
        if not isinstance(body, dict):
            logger.error(f"Search body must be a dict, got {type(body)}: {body}")
            raise ValueError(f"Search body must be a dictionary, got {type(body)}")
        
        # Validation stricte du corps de requête
        try:
            validation_result = self._query_validator.validate_search_query(body)
            if not validation_result.is_valid:
                error_msg = f"Invalid search query: {'; '.join(validation_result.errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Utiliser la version sanitisée si disponible
            if validation_result.sanitized_data:
                body = validation_result.sanitized_data
                
        except ValidationError as e:
            logger.error(f"Query validation failed: {e}")
            raise ValueError(f"Query validation failed: {e}")
        
        # Ajouter size et from_ au body si pas déjà présents
        body = body.copy()  # Éviter de modifier l'original
        if "size" not in body:
            body["size"] = size
        if "from" not in body:
            body["from"] = from_
        
        # Vérifier le cache si activé
        cache_key = None
        if self._query_cache:
            cache_key = self._generate_cache_key(index, body)
            cached_result = await self._query_cache.get(cache_key)
            if cached_result:
                if self._metrics_collector:
                    self._metrics_collector.record_cache_hit("search_query")
                logger.debug(f"Cache hit for search query: {cache_key}")
                return cached_result
        
        # Timer pour métriques
        timer = None
        if self._metrics_collector:
            timer = self._metrics_collector.time_query()
        
        start_time = time.time()
        
        try:
            async def _search():
                async with self.session.post(
                    f"{self.base_url}/{index}/_search",
                    json=body
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Enregistrer le temps Elasticsearch
                        es_time = result.get("took", 0) / 1000.0  # Convertir ms en secondes
                        if timer:
                            timer.record_elasticsearch_time(es_time)
                        
                        return result
                    else:
                        error_text = await response.text()
                        raise ClientError(
                            f"Search failed: HTTP {response.status} - {error_text}",
                            "search",
                            {"index": index, "status": response.status}
                        )
            
            result = await self.execute_with_retry(_search, "search")
            
            # Mettre en cache si activé
            if self._query_cache and cache_key:
                await self._query_cache.set(cache_key, result)
                if self._metrics_collector:
                    self._metrics_collector.record_cache_miss("search_query")
            
            # Enregistrer les métriques
            if timer:
                hits = result.get("hits", {})
                result_count = len(hits.get("hits", []))
                max_score = hits.get("max_score", 0.0) or 0.0
                timer.record_results(result_count, max_score)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Search completed in {elapsed_time:.3f}s, {result.get('hits', {}).get('total', 0)} results")
            
            return result
            
        except Exception as e:
            if timer:
                timer.record_results(-1, 0.0)  # Marquer comme échec
            raise
        finally:
            if timer:
                timer.finish()
    
    async def health(self) -> Dict[str, Any]:
        """
        Vérifie la santé d'Elasticsearch (méthode générique).
        
        Cette méthode est utilisée par les health checks et doit retourner
        un format standard.
        
        Returns:
            Statut de santé du cluster
        """
        async def _health():
            async with self.session.get(
                f"{self.base_url}/_cluster/health"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise ClientError(
                        f"Health check failed: HTTP {response.status} - {error_text}",
                        "health"
                    )
        
        return await self.execute_with_retry(_health, "health")
    
    async def count(
        self,
        index: str,
        body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compte les documents correspondant à une requête (méthode générique).
        
        Cette méthode est utilisée par les moteurs pour compter les documents.
        
        Args:
            index: Nom de l'index
            body: Corps de la requête de comptage
            
        Returns:
            Résultat du comptage
        """
        # Validation défensive
        if body is None:
            logger.error("Count body is None - this indicates a bug in query construction")
            raise ValueError("Count body cannot be None")
        
        if not isinstance(body, dict):
            logger.error(f"Count body must be a dict, got {type(body)}")
            raise ValueError(f"Count body must be a dictionary, got {type(body)}")
        
        async def _count():
            async with self.session.post(
                f"{self.base_url}/{index}/_count",
                json=body
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise ClientError(
                        f"Count failed: HTTP {response.status} - {error_text}",
                        "count",
                        {"index": index, "status": response.status}
                    )
        
        return await self.execute_with_retry(_count, "count")
    
    # ============================================================================
    # MÉTHODES SPÉCIALISÉES POUR LES TRANSACTIONS FINANCIÈRES
    # ============================================================================
    
    async def search_transactions(
        self,
        query: str,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        include_highlights: bool = True,
        include_aggregations: bool = False,
        sort_strategy: SortStrategy = SortStrategy.RELEVANCE,
        query_strategy: QueryStrategy = QueryStrategy.HYBRID
    ) -> Dict[str, Any]:
        """
        Recherche de transactions avec requête optimisée pour le domaine financier.
        
        Args:
            query: Terme de recherche
            user_id: ID de l'utilisateur  
            limit: Nombre de résultats
            offset: Décalage pour pagination
            filters: Filtres additionnels
            include_highlights: Inclure le highlighting
            include_aggregations: Inclure les agrégations
            sort_strategy: Stratégie de tri
            query_strategy: Stratégie de requête
            
        Returns:
            Résultats de recherche formatés
        """
        # Validation des paramètres
        try:
            validated_params = self._validate_search_params(
                query, user_id, limit, offset, filters
            )
        except ValidationError as e:
            raise ValueError(f"Invalid search parameters: {e}")
        
        # Construction de la requête optimisée
        search_body = self._build_financial_search_query(
            validated_params["query"],
            validated_params["user_id"],
            limit,
            offset,
            validated_params["filters"],
            include_highlights,
            include_aggregations,
            sort_strategy,
            query_strategy
        )
        
        # Exécution avec le cache et métriques
        start_time = time.time()
        timer = None
        if self._metrics_collector:
            timer = self._metrics_collector.time_query(query, user_id)
        
        try:
            # Recherche via la méthode générique
            raw_results = await self.search(
                index=self.index_name,
                body=search_body,
                size=limit,
                from_=offset
            )
            
            # Formatage des résultats
            formatted_results = self._result_formatter.format_search_results(raw_results)
            
            # Enregistrer les métriques
            if timer:
                result_count = len(formatted_results.get("hits", []))
                max_score = formatted_results.get("max_score", 0.0)
                timer.record_results(result_count, max_score)
                
                # Ajouter des filtres appliqués aux métriques
                if filters:
                    for filter_name in filters.keys():
                        timer.add_filter(filter_name)
            
            elapsed_time = time.time() - start_time
            logger.info(
                f"Financial search completed: query='{query}', user={user_id}, "
                f"results={len(formatted_results.get('hits', []))}, time={elapsed_time:.3f}s"
            )
            
            return formatted_results
            
        except Exception as e:
            if timer:
                timer.record_results(-1, 0.0)
            logger.error(f"Financial search failed: query='{query}', user={user_id}, error={e}")
            raise
        finally:
            if timer:
                timer.finish()
    
    def _validate_search_params(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Valide les paramètres de recherche."""
        # Valider la requête
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string")
        
        query = query.strip()
        if len(query) > 500:
            raise ValidationError("Query too long (max 500 characters)")
        
        # Valider l'utilisateur
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValidationError("User ID must be a positive integer")
        
        # Valider la pagination
        if not isinstance(limit, int) or limit <= 0:
            raise ValidationError("Limit must be a positive integer")
        if limit > 100:
            raise ValidationError("Limit too large (max 100)")
        
        if not isinstance(offset, int) or offset < 0:
            raise ValidationError("Offset must be a non-negative integer")
        if offset > 10000:
            raise ValidationError("Offset too large (max 10000)")
        
        # Valider et sanitiser les filtres
        validated_filters = {}
        if filters:
            validated_filters = self._filter_validator.validate_and_sanitize(filters)
        
        return {
            "query": query,
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
            "filters": validated_filters
        }
    
    def _build_financial_search_query(
        self,
        query: str,
        user_id: int,
        limit: int,
        offset: int,
        filters: Dict[str, Any],
        include_highlights: bool,
        include_aggregations: bool,
        sort_strategy: SortStrategy,
        query_strategy: QueryStrategy
    ) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch optimisée pour les transactions financières.
        
        Cette méthode utilise le QueryBuilder pour construire des requêtes
        sophistiquées avec toutes les optimisations nécessaires.
        """
        # Utiliser le QueryBuilder pour une construction fluide
        builder = (self._query_builder
                  .reset()
                  .with_user_filter(user_id)
                  .with_text_search(query, query_strategy)
                  .with_filters(filters)
                  .with_sort(sort_strategy)
                  .with_pagination(limit, offset)
                  .with_source_fields()  # Champs par défaut optimisés
                  .with_recency_boost(enabled=True))
        
        if include_highlights:
            builder.with_highlights(enabled=True)
        
        if include_aggregations:
            builder.with_aggregations(
                categories=True,
                merchants=True,
                amounts=True,
                time=True
            )
        
        return builder.build()
    
    async def get_suggestions(
        self,
        partial_query: str,
        user_id: int,
        max_suggestions: int = 10
    ) -> Dict[str, Any]:
        """
        Obtient des suggestions d'auto-complétion optimisées.
        
        Args:
            partial_query: Début de requête
            user_id: ID de l'utilisateur
            max_suggestions: Nombre max de suggestions
            
        Returns:
            Suggestions groupées par type
        """
        if not partial_query or len(partial_query) < 2:
            return {"suggestions": []}
        
        suggestions_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}}
                    ],
                    "should": [
                        {
                            "prefix": {
                                "merchant_name.keyword": {
                                    "value": partial_query,
                                    "boost": 3.0
                                }
                            }
                        },
                        {
                            "match_phrase_prefix": {
                                "primary_description": {
                                    "query": partial_query,
                                    "boost": 2.0
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "merchants": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": max_suggestions,
                        "include": f".*{partial_query}.*"
                    }
                },
                "descriptions": {
                    "terms": {
                        "field": "primary_description.keyword",
                        "size": max_suggestions,
                        "include": f".*{partial_query}.*"
                    }
                }
            },
            "size": 0
        }
        
        async def _get_suggestions():
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_search",
                json=suggestions_query
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise ClientError(
                        f"Suggestions failed: HTTP {response.status} - {error_text}",
                        "suggestions"
                    )
        
        result = await self.execute_with_retry(_get_suggestions, "suggestions")
        
        # Formater les suggestions
        suggestions = []
        
        # Suggestions de marchands
        merchants_agg = result.get("aggregations", {}).get("merchants", {})
        for bucket in merchants_agg.get("buckets", []):
            suggestions.append({
                "type": "merchant",
                "text": bucket["key"],
                "count": bucket["doc_count"]
            })
        
        # Suggestions de descriptions  
        descriptions_agg = result.get("aggregations", {}).get("descriptions", {})
        for bucket in descriptions_agg.get("buckets", []):
            suggestions.append({
                "type": "description", 
                "text": bucket["key"],
                "count": bucket["doc_count"]
            })
        
        # Trier par popularité et limiter
        suggestions.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "suggestions": suggestions[:max_suggestions],
            "query": partial_query,
            "total": len(suggestions)
        }
    
    async def count_transactions(
        self,
        user_id: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Compte le nombre de transactions correspondant aux critères.
        
        Args:
            user_id: ID de l'utilisateur
            filters: Filtres additionnels
            
        Returns:
            Nombre de transactions
        """
        # Validation des paramètres
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("User ID must be a positive integer")
        
        # Valider les filtres
        validated_filters = {}
        if filters:
            try:
                validated_filters = self._filter_validator.validate_and_sanitize(filters)
            except ValidationError as e:
                raise ValueError(f"Invalid filters: {e}")
        
        # Construction de la requête de comptage
        count_query = {
            "query": {
                "bool": {
                    "must": [{"term": {"user_id": user_id}}]
                }
            }
        }
        
        # Ajouter les filtres via les helpers
        if validated_filters:
            count_query = ElasticsearchHelpers.add_filters_to_query(count_query, validated_filters)
        
        try:
            result = await self.count(self.index_name, count_query)
            return result.get("count", 0)
        except Exception as e:
            logger.error(f"Count transactions failed: user={user_id}, error={e}")
            raise ClientError(f"Failed to count transactions: {e}", "count_transactions")
    
    async def get_transaction_stats(
        self,
        user_id: int,
        filters: Optional[Dict[str, Any]] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtient des statistiques détaillées sur les transactions.
        
        Args:
            user_id: ID de l'utilisateur
            filters: Filtres additionnels
            period_days: Période d'analyse en jours
            
        Returns:
            Statistiques complètes
        """
        # Construction de la requête avec agrégations
        stats_query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                        {
                            "range": {
                                "transaction_date": {
                                    "gte": f"now-{period_days}d/d"
                                }
                            }
                        }
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "total_amount": {
                    "sum": {"field": "amount"}
                },
                "avg_amount": {
                    "avg": {"field": "amount"}
                },
                "amount_stats": {
                    "stats": {"field": "amount"}
                },
                "transaction_types": {
                    "terms": {
                        "field": "transaction_type",
                        "size": 10
                    }
                },
                "top_merchants": {
                    "terms": {
                        "field": "merchant_name.keyword",
                        "size": 10
                    }
                },
                "daily_transactions": {
                    "date_histogram": {
                        "field": "transaction_date",
                        "calendar_interval": "day"
                    }
                },
                "amount_ranges": {
                    "range": {
                        "field": "amount",
                        "ranges": [
                            {"key": "0-10", "from": 0, "to": 10},
                            {"key": "10-50", "from": 10, "to": 50},
                            {"key": "50-100", "from": 50, "to": 100},
                            {"key": "100-500", "from": 100, "to": 500},
                            {"key": "500+", "from": 500}
                        ]
                    }
                }
            }
        }
        
        # Ajouter les filtres si fournis
        if filters:
            try:
                validated_filters = self._filter_validator.validate_and_sanitize(filters)
                stats_query = ElasticsearchHelpers.add_filters_to_query(stats_query, validated_filters)
            except ValidationError as e:
                raise ValueError(f"Invalid filters: {e}")
        
        try:
            result = await self.search(self.index_name, stats_query)
            
            # Formater les statistiques
            aggs = result.get("aggregations", {})
            amount_stats = aggs.get("amount_stats", {})
            
            return {
                "period_days": period_days,
                "total_transactions": result.get("hits", {}).get("total", 0),
                "amount_statistics": {
                    "total": aggs.get("total_amount", {}).get("value", 0),
                    "average": aggs.get("avg_amount", {}).get("value", 0),
                    "min": amount_stats.get("min", 0),
                    "max": amount_stats.get("max", 0),
                    "count": amount_stats.get("count", 0)
                },
                "transaction_types": [
                    {"type": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in aggs.get("transaction_types", {}).get("buckets", [])
                ],
                "top_merchants": [
                    {"merchant": bucket["key"], "count": bucket["doc_count"]}
                    for bucket in aggs.get("top_merchants", {}).get("buckets", [])
                ],
                "daily_activity": [
                    {
                        "date": bucket["key_as_string"],
                        "count": bucket["doc_count"]
                    }
                    for bucket in aggs.get("daily_transactions", {}).get("buckets", [])
                ],
                "amount_distribution": [
                    {
                        "range": bucket["key"],
                        "count": bucket["doc_count"]
                    }
                    for bucket in aggs.get("amount_ranges", {}).get("buckets", [])
                ]
            }
            
        except Exception as e:
            logger.error(f"Get transaction stats failed: user={user_id}, error={e}")
            raise ClientError(f"Failed to get transaction statistics: {e}", "get_stats")
    
    async def get_index_info(self) -> Dict[str, Any]:
        """
        Obtient des informations détaillées sur l'index.
        
        Returns:
            Informations sur l'index et ses performances
        """
        try:
            # Informations de base sur l'index
            async with self.session.get(f"{self.base_url}/{self.index_name}") as response:
                if response.status != 200:
                    raise ClientError(f"Index info request failed: HTTP {response.status}")
                
                index_info = await response.json()
            
            # Statistiques détaillées
            stats = await self._get_basic_index_stats()
            
            # Mapping de l'index
            async with self.session.get(f"{self.base_url}/{self.index_name}/_mapping") as response:
                mapping_info = {}
                if response.status == 200:
                    mapping_data = await response.json()
                    mapping_info = mapping_data.get(self.index_name, {}).get("mappings", {})
            
            # Paramètres de l'index
            async with self.session.get(f"{self.base_url}/{self.index_name}/_settings") as response:
                settings_info = {}
                if response.status == 200:
                    settings_data = await response.json()
                    settings_info = settings_data.get(self.index_name, {}).get("settings", {})
            
            return {
                "index_name": self.index_name,
                "creation_date": index_info.get(self.index_name, {}).get("settings", {}).get("index", {}).get("creation_date"),
                "uuid": index_info.get(self.index_name, {}).get("settings", {}).get("index", {}).get("uuid"),
                "stats": stats,
                "mapping": mapping_info,
                "settings": settings_info,
                "cache_info": await self._get_cache_info() if self._query_cache else {},
                "metrics": self._metrics_collector.get_summary() if self._metrics_collector else {}
            }
            
        except Exception as e:
            logger.error(f"Get index info failed: {e}")
            raise ClientError(f"Failed to get index information: {e}", "get_index_info")
    
    async def _get_cache_info(self) -> Dict[str, Any]:
        """Obtient les informations du cache."""
        if not self._query_cache:
            return {}
        
        return await self._query_cache.get_info()
    
    def _generate_cache_key(self, index: str, body: Dict[str, Any]) -> str:
        """Génère une clé de cache pour une requête."""
        import hashlib
        
        # Créer une représentation stable de la requête
        cache_data = {
            "index": index,
            "query": body.get("query", {}),
            "size": body.get("size", 20),
            "from": body.get("from", 0),
            "sort": body.get("sort", [])
        }
        
        # Sérialiser de manière déterministe
        cache_str = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
        
        # Hasher pour créer une clé courte
        hash_obj = hashlib.md5(cache_str.encode())
        return f"es_query:{hash_obj.hexdigest()}"
    
    # ============================================================================
    # MÉTHODES DE MAINTENANCE ET OPTIMISATION
    # ============================================================================
    
    async def refresh_index(self) -> bool:
        """
        Force un refresh de l'index pour rendre les changements visibles.
        
        Returns:
            True si le refresh a réussi
        """
        try:
            async with self.session.post(f"{self.base_url}/{self.index_name}/_refresh") as response:
                if response.status == 200:
                    logger.info(f"Index {self.index_name} refreshed successfully")
                    return True
                else:
                    logger.warning(f"Index refresh failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Index refresh error: {e}")
            return False
    
    async def clear_cache(self) -> bool:
        """
        Vide le cache des requêtes.
        
        Returns:
            True si le cache a été vidé
        """
        if self._query_cache:
            try:
                await self._query_cache.clear()
                logger.info("Query cache cleared successfully")
                return True
            except Exception as e:
                logger.error(f"Cache clear error: {e}")
                return False
        return True
    
    async def optimize_index(self) -> bool:
        """
        Lance une optimisation de l'index (force merge).
        
        Returns:
            True si l'optimisation a été lancée
        """
        try:
            # Force merge avec max 1 segment pour optimisation maximale
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_forcemerge",
                params={"max_num_segments": 1}
            ) as response:
                if response.status == 200:
                    logger.info(f"Index {self.index_name} optimization started")
                    return True
                else:
                    logger.warning(f"Index optimization failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Index optimization error: {e}")
            return False
    
    async def validate_query_performance(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et analyse les performances d'une requête sans l'exécuter.
        
        Args:
            body: Corps de la requête à valider
            
        Returns:
            Analyse des performances de la requête
        """
        try:
            # Utiliser l'API _validate/query pour vérifier la requête
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_validate/query",
                json={"query": body.get("query", {})},
                params={"explain": "true"}
            ) as response:
                
                if response.status == 200:
                    validation_result = await response.json()
                    
                    # Analyser la complexité de la requête
                    complexity = self._query_validator._analyze_query_complexity(body)
                    
                    return {
                        "valid": validation_result.get("valid", False),
                        "explanations": validation_result.get("explanations", []),
                        "complexity_score": complexity.score,
                        "complexity_analysis": {
                            "nested_depth": complexity.nested_depth,
                            "bool_clauses": complexity.bool_clauses,
                            "wildcard_count": complexity.wildcard_count,
                            "regexp_count": complexity.regexp_count,
                            "is_complex": complexity.is_complex
                        },
                        "recommendations": self._get_performance_recommendations(complexity)
                    }
                else:
                    error_text = await response.text()
                    return {
                        "valid": False,
                        "error": f"Validation failed: HTTP {response.status} - {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Query validation error: {e}")
            return {
                "valid": False,
                "error": f"Validation error: {e}"
            }
    
    def _get_performance_recommendations(self, complexity) -> List[str]:
        """Génère des recommandations de performance basées sur la complexité."""
        recommendations = []
        
        if complexity.is_complex:
            recommendations.append("Query is complex and may impact performance")
        
        if complexity.wildcard_count > 5:
            recommendations.append("Consider reducing wildcard queries for better performance")
        
        if complexity.regexp_count > 0:
            recommendations.append("Regular expressions are expensive, consider alternatives")
        
        if complexity.bool_clauses > 20:
            recommendations.append("Too many boolean clauses, consider simplifying the query")
        
        if complexity.nested_depth > 5:
            recommendations.append("Query nesting is deep, consider flattening the structure")
        
        return recommendations
    
    # ============================================================================
    # MÉTHODES DE MONITORING ET MÉTRIQUES
    # ============================================================================
    
    def get_client_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques complètes du client.
        
        Returns:
            Métriques du client, cache et Elasticsearch
        """
        metrics = super().get_metrics()
        
        # Ajouter les métriques spécifiques Elasticsearch
        if self._metrics_collector:
            search_metrics = self._metrics_collector.get_summary()
            metrics["search_service"] = search_metrics
        
        # Ajouter les métriques du cache
        if self._query_cache:
            cache_stats = self._query_cache.get_stats()
            metrics["cache"] = {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate,
                "size": cache_stats.size,
                "memory_usage_mb": cache_stats.memory_usage_bytes / 1024 / 1024
            }
        
        return metrics
    
    async def export_metrics(self, format: str = "prometheus") -> str:
        """
        Exporte les métriques dans le format spécifié.
        
        Args:
            format: Format d'export ("prometheus" ou "json")
            
        Returns:
            Métriques formatées
        """
        if not self._metrics_collector:
            return ""
        
        if format.lower() == "prometheus":
            return self._metrics_collector.export_prometheus()
        elif format.lower() == "json":
            import json
            return json.dumps(self._metrics_collector.get_summary(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # ============================================================================
    # MÉTHODES DE CONTEXTE ET NETTOYAGE
    # ============================================================================
    
    async def __aenter__(self):
        """Support du context manager."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique."""
        await self.stop()
    
    def __repr__(self) -> str:
        """Représentation string du client."""
        return (
            f"ElasticsearchClient("
            f"url='{self.base_url}', "
            f"index='{self.index_name}', "
            f"timeout={self.timeout}, "
            f"cache_enabled={self.enable_cache}, "
            f"metrics_enabled={self.enable_metrics}"
            f")"
        )