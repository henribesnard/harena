"""
Client Elasticsearch optimisé pour Bonsai - Search Service
Spécialisé dans les recherches lexicales de transactions financières
"""

import logging
import ssl
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import aiohttp

from .base_client import (
    BaseClient, 
    RetryConfig, 
    CircuitBreakerConfig, 
    HealthCheckConfig,
    ClientStatus
)
from config import settings, INDEXED_FIELDS

logger = logging.getLogger(__name__)


class ElasticsearchClient(BaseClient):
    """
    Client Elasticsearch/Bonsai optimisé pour le Search Service
    
    Responsabilités principales:
    - Recherches lexicales ultra-rapides (<50ms)
    - Validation défensive des requêtes (évite body=None)
    - Gestion SSL native pour Bonsai
    - Interface générique pour lexical_engine.py
    - Métriques spécialisées Elasticsearch
    - Health checks détaillés (cluster, index, mapping)
    
    Optimisations spécifiques:
    - Connection pooling pour Bonsai
    - Query validation avant envoi
    - Gestion des timeouts adaptatifs
    - Cache des requêtes fréquentes
    """
    
    def __init__(
        self,
        bonsai_url: str,
        index_name: str = "harena_transactions",
        timeout: float = 10.0,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ):
        # Configuration SSL spécialisée pour Bonsai
        health_check_config = HealthCheckConfig(
            enabled=True,
            interval_seconds=30.0,
            timeout_seconds=5.0,
            endpoint="/"
        )
        
        # Headers spécialisés Elasticsearch
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **kwargs.get("headers", {})
        }
        
        super().__init__(
            base_url=bonsai_url,
            service_name="elasticsearch",
            timeout=timeout,
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            health_check_config=health_check_config,
            headers=headers
        )
        
        self.index_name = index_name
        self.ssl_context = ssl.create_default_context()
        
        # Cache des requêtes fréquentes (LRU simple)
        self._query_cache: Dict[str, Dict] = {}
        self._max_cache_size = 100
        
        # Statistiques spécialisées Elasticsearch
        self.slow_query_threshold = 1.0  # 1s considéré lent pour search
        self.query_stats = {
            "search_count": 0,
            "count_count": 0,
            "aggregation_count": 0,
            "cache_hits": 0
        }
        
        logger.info(f"Elasticsearch client initialized for Bonsai index: {index_name}")
    
    async def start(self):
        """Démarre le client avec configuration SSL optimisée pour Bonsai"""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            # Connector SSL optimisé pour Bonsai
            connector = aiohttp.TCPConnector(
                ssl=self.ssl_context,
                limit=20,           # Pool de connexions
                limit_per_host=10,  # Connexions par host
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self.headers
            )
            
            logger.info(f"{self.service_name} client started with SSL for Bonsai")
            
            # Test initial de connexion
            try:
                await self._verify_bonsai_connection()
            except Exception as e:
                logger.warning(f"Initial Bonsai connection test failed: {e}")
    
    async def _verify_bonsai_connection(self):
        """Vérifie la connexion initiale à Bonsai"""
        async with self.session.get(self.base_url) as response:
            if response.status == 200:
                cluster_info = await response.json()
                cluster_name = cluster_info.get("cluster_name", "unknown")
                version = cluster_info.get("version", {}).get("number", "unknown")
                logger.info(f"✅ Bonsai connected: {cluster_name} elasticsearch v{version}")
            else:
                logger.warning(f"⚠️ Bonsai responded with status {response.status}")
    
    async def test_connection(self) -> bool:
        """Teste la connectivité de base à Bonsai Elasticsearch"""
        try:
            async def _test():
                async with self.session.get(self.base_url) as response:
                    return response.status == 200
            
            result = await self.execute_with_retry(_test, "connection_test")
            if result:
                logger.debug("Bonsai connection test successful")
            else:
                logger.warning("Bonsai connection test failed")
            return result
            
        except Exception as e:
            logger.error(f"Bonsai connection test failed: {e}")
            return False
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Effectue une vérification de santé complète de Bonsai"""
        health_info = {
            "status": "unknown",
            "cluster_name": None,
            "version": None,
            "index_exists": False,
            "index_health": None,
            "mapping_valid": False,
            "test_user_transactions": 0
        }
        
        try:
            # 1. Vérifier le cluster Elasticsearch
            async with self.session.get(self.base_url) as response:
                if response.status == 200:
                    cluster_info = await response.json()
                    health_info.update({
                        "cluster_name": cluster_info.get("cluster_name", "unknown"),
                        "version": cluster_info.get("version", {}).get("number", "unknown")
                    })
                else:
                    health_info["status"] = "unhealthy"
                    health_info["error"] = f"Cluster HTTP {response.status}"
                    return health_info
            
            # 2. Vérifier l'existence de l'index
            async with self.session.head(f"{self.base_url}/{self.index_name}") as response:
                health_info["index_exists"] = response.status == 200
            
            if not health_info["index_exists"]:
                health_info["status"] = "degraded"
                health_info["error"] = f"Index {self.index_name} does not exist"
                return health_info
            
            # 3. Vérifier la santé de l'index
            async with self.session.get(
                f"{self.base_url}/_cluster/health/{self.index_name}"
            ) as response:
                if response.status == 200:
                    index_health = await response.json()
                    health_info["index_health"] = index_health.get("status", "unknown")
                
            # 4. Vérifier le mapping (champs critiques)
            await self._check_mapping_validity(health_info)
            
            # 5. Compter les transactions de test (si configuré)
            if hasattr(settings, 'test_user_id'):
                count = await self._count_test_transactions()
                health_info["test_user_transactions"] = count
            
            # Déterminer le statut final
            if (health_info["index_exists"] and 
                health_info["mapping_valid"] and 
                health_info["index_health"] in ["green", "yellow"]):
                health_info["status"] = "healthy"
            elif health_info["index_exists"]:
                health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"
            
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.error(f"Elasticsearch health check failed: {e}")
        
        return health_info
    
    async def _check_mapping_validity(self, health_info: Dict[str, Any]):
        """Vérifie que le mapping contient les champs critiques"""
        try:
            async with self.session.get(
                f"{self.base_url}/{self.index_name}/_mapping"
            ) as response:
                if response.status == 200:
                    mapping_data = await response.json()
                    properties = mapping_data.get(
                        self.index_name, {}
                    ).get("mappings", {}).get("properties", {})
                    
                    # Champs critiques pour le Search Service
                    required_fields = [
                        "user_id", "searchable_text", "primary_description", 
                        "merchant_name", "amount", "date"
                    ]
                    
                    missing_fields = [
                        field for field in required_fields 
                        if field not in properties
                    ]
                    
                    health_info["mapping_valid"] = len(missing_fields) == 0
                    if missing_fields:
                        health_info["missing_fields"] = missing_fields
                        logger.warning(f"Missing critical fields in mapping: {missing_fields}")
                    
        except Exception as e:
            logger.warning(f"Could not check mapping validity: {e}")
            health_info["mapping_valid"] = False
    
    async def _count_test_transactions(self) -> int:
        """Compte les transactions de test pour vérifier les données"""
        try:
            test_user_id = getattr(settings, 'test_user_id', 34)
            count_query = {"query": {"term": {"user_id": test_user_id}}}
            
            async with self.session.post(
                f"{self.base_url}/{self.index_name}/_count",
                json=count_query
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("count", 0)
        except Exception as e:
            logger.debug(f"Could not count test transactions: {e}")
        
        return 0
    
    # ============================================================================
    # INTERFACE GÉNÉRIQUE POUR LEXICAL_ENGINE
    # ============================================================================
    
    async def search(
        self,
        index: str,
        body: Optional[Dict[str, Any]] = None,
        size: int = 20,
        from_: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Interface générique de recherche pour lexical_engine.py
        
        VALIDATION DÉFENSIVE CRITIQUE:
        Cette méthode est appelée par lexical_engine et doit absolument
        valider que body n'est pas None pour éviter les erreurs silencieuses
        
        Args:
            index: Nom de l'index Elasticsearch
            body: Corps de la requête Elasticsearch (OBLIGATOIRE)
            size: Nombre de résultats à retourner
            from_: Offset pour pagination
            **kwargs: Paramètres additionnels Elasticsearch
            
        Returns:
            Réponse Elasticsearch formatée
            
        Raises:
            ValueError: Si body est None ou invalide
            Exception: Si la recherche échoue
        """
        # 🚨 VALIDATION DÉFENSIVE CRITIQUE
        if body is None:
            error_msg = (
                f"Search body cannot be None for index '{index}'. "
                f"This indicates a bug in query construction. "
                f"Check the calling code in lexical_engine.py"
            )
            logger.error(error_msg)
            logger.error(f"Search called with: index={index}, size={size}, from_={from_}")
            raise ValueError(error_msg)
        
        if not isinstance(body, dict):
            error_msg = f"Search body must be a dict, got {type(body)}: {body}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validation de l'index
        if not index:
            raise ValueError("Index name cannot be empty")
        
        # Ajouter size et from_ au corps si pas déjà présents
        search_body = body.copy()
        if "size" not in search_body:
            search_body["size"] = size
        if "from" not in search_body:
            search_body["from"] = from_
        
        # Vérifier le cache si activé
        cache_key = self._generate_cache_key(index, search_body)
        if cache_key in self._query_cache:
            self.query_stats["cache_hits"] += 1
            logger.debug(f"Cache hit for search query")
            return self._query_cache[cache_key]
        
        async def _search():
            async with self.session.post(
                f"{self.base_url}/{index}/_search",
                json=search_body
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Ajouter au cache si la requête a réussi
                    self._add_to_cache(cache_key, result)
                    
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Elasticsearch search failed: HTTP {response.status} - {error_text}"
                    )
        
        # Exécuter avec retry et circuit breaker
        result = await self.execute_with_retry(_search, "search")
        
        # Statistiques
        self.query_stats["search_count"] += 1
        
        return result
    
    async def count(
        self,
        index: str,
        body: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Interface générique de comptage pour lexical_engine.py
        
        Args:
            index: Nom de l'index Elasticsearch
            body: Corps de la requête de comptage (peut être None pour count total)
            **kwargs: Paramètres additionnels
            
        Returns:
            Résultat du comptage Elasticsearch
        """
        # Pour count, body peut être None (count total de l'index)
        if body is not None and not isinstance(body, dict):
            error_msg = f"Count body must be a dict or None, got {type(body)}: {body}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Si body est None, créer une query match_all
        count_body = body or {"query": {"match_all": {}}}
        
        async def _count():
            async with self.session.post(
                f"{self.base_url}/{index}/_count",
                json=count_body
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Elasticsearch count failed: HTTP {response.status} - {error_text}"
                    )
        
        result = await self.execute_with_retry(_count, "count")
        
        # Statistiques
        self.query_stats["count_count"] += 1
        
        return result
    
    async def health(self) -> Dict[str, Any]:
        """
        Interface générique de santé pour les health checks
        
        Returns:
            Statut de santé du cluster Elasticsearch
        """
        async def _health():
            async with self.session.get(
                f"{self.base_url}/_cluster/health"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Elasticsearch health failed: HTTP {response.status} - {error_text}"
                    )
        
        return await self.execute_with_retry(_health, "health")
    
    # ============================================================================
    # MÉTHODES SPÉCIALISÉES SEARCH SERVICE
    # ============================================================================
    
    async def search_transactions(
        self,
        user_id: int,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        include_highlights: bool = False,
        sort_by: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Recherche de transactions optimisée pour le domaine financier
        
        Args:
            user_id: ID utilisateur (obligatoire pour sécurité)
            query: Terme de recherche textuelle (optionnel)
            filters: Filtres additionnels (catégorie, montant, date)
            limit: Nombre de résultats
            offset: Décalage pagination
            include_highlights: Inclure surlignage des termes
            sort_by: Critères de tri personnalisés
            
        Returns:
            Résultats de recherche formatés
        """
        search_body = self._build_transaction_search_query(
            user_id=user_id,
            query=query,
            filters=filters,
            limit=limit,
            offset=offset,
            include_highlights=include_highlights,
            sort_by=sort_by
        )
        
        return await self.search(
            index=self.index_name,
            body=search_body,
            size=limit,
            from_=offset
        )
    
    def _build_transaction_search_query(
        self,
        user_id: int,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        include_highlights: bool = False,
        sort_by: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Construit une requête Elasticsearch optimisée pour les transactions
        
        Optimisations appliquées:
        - user_id filter obligatoire (sécurité)
        - Multi-match avec boost sur champs importants
        - Tri par pertinence puis date
        - Highlighting intelligent si demandé
        """
        # Requête de base avec filtre utilisateur obligatoire
        bool_query = {
            "bool": {
                "must": [
                    {"term": {"user_id": user_id}}
                ]
            }
        }
        
        # Ajouter recherche textuelle si fournie
        if query and query.strip():
            text_query = {
                "multi_match": {
                    "query": query.strip(),
                    "fields": [
                        "searchable_text^2.0",
                        "primary_description^1.5", 
                        "merchant_name^1.8",
                        "category_name^1.2"
                    ],
                    "type": "best_fields",
                    "operator": "or",
                    "fuzziness": "AUTO"
                }
            }
            bool_query["bool"]["must"].append(text_query)
        
        # Ajouter filtres additionnels
        if filters:
            self._apply_transaction_filters(bool_query, filters)
        
        # Corps de la requête complet
        search_body = {
            "query": bool_query,
            "size": limit,
            "from": offset,
            "_source": [
                "transaction_id", "user_id", "account_id",
                "amount", "amount_abs", "transaction_type", "currency_code",
                "date", "primary_description", "merchant_name", "category_name", 
                "operation_type", "month_year", "weekday"
            ]
        }
        
        # Tri (par défaut: pertinence puis date décroissante)
        if sort_by:
            search_body["sort"] = sort_by
        else:
            search_body["sort"] = [
                {"_score": {"order": "desc"}},
                {"date": {"order": "desc", "unmapped_type": "date"}}
            ]
        
        # Highlighting si demandé
        if include_highlights and query:
            search_body["highlight"] = {
                "fields": {
                    "searchable_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 2,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    },
                    "primary_description": {
                        "fragment_size": 100,
                        "number_of_fragments": 1,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    },
                    "merchant_name": {
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            }
        
        return search_body
    
    def _apply_transaction_filters(self, bool_query: Dict[str, Any], filters: Dict[str, Any]):
        """Applique les filtres spécifiques aux transactions"""
        must_filters = bool_query["bool"]["must"]
        
        # Filtre de catégorie
        if "category" in filters and filters["category"]:
            must_filters.append({
                "term": {"category_name.keyword": filters["category"]}
            })
        
        # Filtre de marchand
        if "merchant" in filters and filters["merchant"]:
            must_filters.append({
                "term": {"merchant_name.keyword": filters["merchant"]}
            })
        
        # Filtre de montant (plage)
        if "amount_min" in filters or "amount_max" in filters:
            amount_filter = {"range": {"amount_abs": {}}}
            if "amount_min" in filters:
                amount_filter["range"]["amount_abs"]["gte"] = filters["amount_min"]
            if "amount_max" in filters:
                amount_filter["range"]["amount_abs"]["lte"] = filters["amount_max"]
            must_filters.append(amount_filter)
        
        # Filtre de date (plage)
        if "date_from" in filters or "date_to" in filters:
            date_filter = {"range": {"date": {}}}
            if "date_from" in filters:
                date_filter["range"]["date"]["gte"] = filters["date_from"]
            if "date_to" in filters:
                date_filter["range"]["date"]["lte"] = filters["date_to"]
            must_filters.append(date_filter)
        
        # Filtre de type de transaction
        if "transaction_type" in filters and filters["transaction_type"] != "all":
            must_filters.append({
                "term": {"transaction_type": filters["transaction_type"]}
            })
    
    async def aggregate_transactions(
        self,
        user_id: int,
        aggregation_type: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Effectue des agrégations sur les transactions
        
        Args:
            user_id: ID utilisateur
            aggregation_type: Type d'agrégation (by_category, by_merchant, by_month)
            filters: Filtres à appliquer avant agrégation
            size: Nombre de buckets à retourner
            
        Returns:
            Résultats d'agrégation
        """
        agg_query = {
            "query": {
                "bool": {
                    "must": [{"term": {"user_id": user_id}}]
                }
            },
            "size": 0,  # Ne pas retourner les documents
            "aggs": {}
        }
        
        # Ajouter filtres si fournis
        if filters:
            self._apply_transaction_filters(agg_query, filters)
        
        # Configurer l'agrégation selon le type
        if aggregation_type == "by_category":
            agg_query["aggs"]["categories"] = {
                "terms": {
                    "field": "category_name.keyword",
                    "size": size
                },
                "aggs": {
                    "total_amount": {"sum": {"field": "amount_abs"}},
                    "avg_amount": {"avg": {"field": "amount_abs"}}
                }
            }
        elif aggregation_type == "by_merchant":
            agg_query["aggs"]["merchants"] = {
                "terms": {
                    "field": "merchant_name.keyword", 
                    "size": size
                },
                "aggs": {
                    "total_amount": {"sum": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}}
                }
            }
        elif aggregation_type == "by_month":
            agg_query["aggs"]["months"] = {
                "terms": {
                    "field": "month_year",
                    "size": size,
                    "order": {"_key": "desc"}
                },
                "aggs": {
                    "total_amount": {"sum": {"field": "amount_abs"}},
                    "transaction_count": {"value_count": {"field": "transaction_id"}}
                }
            }
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        
        result = await self.search(
            index=self.index_name,
            body=agg_query
        )
        
        # Statistiques
        self.query_stats["aggregation_count"] += 1
        
        return result
    
    # ============================================================================
    # CACHE ET OPTIMISATIONS
    # ============================================================================
    
    def _generate_cache_key(self, index: str, body: Dict[str, Any]) -> str:
        """Génère une clé de cache pour une requête"""
        import hashlib
        import json
        
        # Créer une représentation stable de la requête
        cache_data = {
            "index": index,
            "body": body
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return f"es_query:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Ajoute un résultat au cache avec gestion LRU simple"""
        if len(self._query_cache) >= self._max_cache_size:
            # Supprimer le plus ancien (LRU simple)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = result
    
    def clear_cache(self):
        """Vide le cache des requêtes"""
        self._query_cache.clear()
        logger.info("Elasticsearch query cache cleared")
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques spécialisées Elasticsearch"""
        base_metrics = self.get_metrics()
        
        return {
            **base_metrics,
            "elasticsearch_stats": {
                **self.query_stats,
                "cache_size": len(self._query_cache),
                "cache_hit_rate": (
                    self.query_stats["cache_hits"] / max(self.query_stats["search_count"], 1)
                )
            }
        }
    
    def reset_query_stats(self):
        """Remet à zéro les statistiques de requêtes"""
        self.query_stats = {
            "search_count": 0,
            "count_count": 0, 
            "aggregation_count": 0,
            "cache_hits": 0
        }
        self.reset_metrics()
        logger.info("Elasticsearch query stats reset")


# === EXPORTS ===

__all__ = [
    "ElasticsearchClient"
]