"""
Client Elasticsearch optimisé pour le Search Service.

Ce client encapsule toutes les interactions avec Elasticsearch,
avec une configuration centralisée et des optimisations de performance.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import json

import aiohttp
from elasticsearch import AsyncElasticsearch, exceptions as es_exceptions
from elasticsearch.helpers import async_bulk

from ..config.settings import SearchServiceSettings, get_settings
from ..models.service_contracts import SearchServiceError
from .base_client import BaseClient


logger = logging.getLogger(__name__)


class ElasticsearchConnectionError(Exception):
    """Erreur de connexion Elasticsearch."""
    pass


class ElasticsearchQueryError(Exception):
    """Erreur de requête Elasticsearch."""
    pass


class ElasticsearchClient(BaseClient):
    """
    Client Elasticsearch optimisé pour la recherche financière.
    
    Fonctionnalités:
    - Connexion asynchrone haute performance
    - Gestion automatique des erreurs et retry
    - Optimisations pour les requêtes financières
    - Health checking et monitoring
    - Configuration centralisée
    """
    
    def __init__(self, settings: Optional[SearchServiceSettings] = None):
        self.settings = settings or get_settings()
        self.client: Optional[AsyncElasticsearch] = None
        self.index_name = self.settings.ELASTICSEARCH_INDEX
        
        # Métriques de performance
        self.query_count = 0
        self.total_query_time = 0.0
        self.error_count = 0
        self.last_health_check = None
        self.is_healthy = False
        
        # Circuit breaker
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_open = False
        
        logger.info(f"Elasticsearch client initialized for index: {self.index_name}")
    
    async def connect(self) -> None:
        """
        Initialise la connexion Elasticsearch.
        
        Raises:
            ElasticsearchConnectionError: Si la connexion échoue
        """
        try:
            self.client = AsyncElasticsearch(
                hosts=[self.settings.elasticsearch_url],
                timeout=self.settings.ELASTICSEARCH_TIMEOUT,
                max_retries=self.settings.ELASTICSEARCH_MAX_RETRIES,
                retry_on_timeout=self.settings.ELASTICSEARCH_RETRY_ON_TIMEOUT,
                # Optimisations de performance
                sniff_on_start=True,
                sniff_on_connection_fail=True,
                sniffer_timeout=60,
                # Configuration SSL si nécessaire
                verify_certs=self.settings.is_production,
                ssl_show_warn=False
            )
            
            # Test de connexion
            await self.health_check()
            
            if not self.is_healthy:
                raise ElasticsearchConnectionError("Elasticsearch n'est pas accessible")
            
            logger.info("✅ Connexion Elasticsearch établie")
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion Elasticsearch: {str(e)}")
            raise ElasticsearchConnectionError(f"Impossible de se connecter à Elasticsearch: {str(e)}")
    
    async def disconnect(self) -> None:
        """Ferme la connexion Elasticsearch."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("🔌 Connexion Elasticsearch fermée")
    
    async def health_check(self) -> bool:
        """
        Vérifie la santé du cluster Elasticsearch.
        
        Returns:
            bool: True si le cluster est sain
        """
        try:
            if not self.client:
                return False
            
            # Vérification cluster
            health = await self.client.cluster.health()
            cluster_status = health.get("status", "red")
            
            # Vérification index
            index_exists = await self.client.indices.exists(index=self.index_name)
            
            self.is_healthy = cluster_status in ["green", "yellow"] and index_exists
            self.last_health_check = datetime.utcnow()
            
            if self.is_healthy:
                # Reset circuit breaker si la santé est OK
                self.circuit_breaker_failures = 0
                self.circuit_breaker_open = False
                logger.debug(f"✅ Elasticsearch healthy - Status: {cluster_status}")
            else:
                logger.warning(f"⚠️ Elasticsearch unhealthy - Status: {cluster_status}, Index exists: {index_exists}")
            
            return self.is_healthy
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            self.is_healthy = False
            self._handle_circuit_breaker()
            return False
    
    def _handle_circuit_breaker(self) -> None:
        """Gère le circuit breaker en cas d'erreurs répétées."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = datetime.utcnow()
        
        if self.circuit_breaker_failures >= 5:
            self.circuit_breaker_open = True
            logger.warning("🚨 Circuit breaker ouvert - Trop d'erreurs Elasticsearch")
    
    def _should_skip_request(self) -> bool:
        """Détermine si la requête doit être ignorée à cause du circuit breaker."""
        if not self.circuit_breaker_open:
            return False
        
        # Réessayer après 1 minute
        if (self.circuit_breaker_last_failure and 
            datetime.utcnow() - self.circuit_breaker_last_failure > timedelta(minutes=1)):
            self.circuit_breaker_open = False
            return False
        
        return True
    
    async def search(
        self,
        query: Dict[str, Any],
        size: int = 20,
        from_: int = 0,
        timeout: str = "5s",
        preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Exécute une requête de recherche Elasticsearch.
        
        Args:
            query: Requête Elasticsearch
            size: Nombre de résultats à retourner
            from_: Offset pour la pagination
            timeout: Timeout de la requête
            preference: Préférence de routage
            
        Returns:
            Dict: Résultats Elasticsearch
            
        Raises:
            ElasticsearchQueryError: Si la requête échoue
        """
        if self._should_skip_request():
            raise ElasticsearchQueryError("Circuit breaker ouvert")
        
        if not self.client:
            raise ElasticsearchQueryError("Client Elasticsearch non connecté")
        
        start_time = datetime.utcnow()
        
        try:
            # Optimisations pour les requêtes financières
            search_params = {
                "index": self.index_name,
                "body": query,
                "size": size,
                "from": from_,
                "timeout": timeout,
                "track_total_hits": True,
                "request_cache": True  # Cache pour les requêtes répétées
            }
            
            # Préférence de routage pour performance
            if preference:
                search_params["preference"] = preference
            
            # Exécution de la requête
            response = await self.client.search(**search_params)
            
            # Métriques
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.query_count += 1
            self.total_query_time += execution_time
            
            logger.debug(f"✅ Requête ES exécutée en {execution_time:.2f}ms")
            
            return response
            
        except es_exceptions.RequestError as e:
            self.error_count += 1
            error_msg = f"Erreur de requête Elasticsearch: {str(e)}"
            logger.error(error_msg)
            raise ElasticsearchQueryError(error_msg)
            
        except es_exceptions.ConnectionError as e:
            self.error_count += 1
            self._handle_circuit_breaker()
            error_msg = f"Erreur de connexion Elasticsearch: {str(e)}"
            logger.error(error_msg)
            raise ElasticsearchConnectionError(error_msg)
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"Erreur inattendue Elasticsearch: {str(e)}"
            logger.error(error_msg)
            raise ElasticsearchQueryError(error_msg)
    
    async def count(self, query: Dict[str, Any]) -> int:
        """
        Compte le nombre de documents correspondant à une requête.
        
        Args:
            query: Requête Elasticsearch
            
        Returns:
            int: Nombre de documents
        """
        if self._should_skip_request():
            raise ElasticsearchQueryError("Circuit breaker ouvert")
        
        if not self.client:
            raise ElasticsearchQueryError("Client Elasticsearch non connecté")
        
        try:
            response = await self.client.count(
                index=self.index_name,
                body=query
            )
            
            return response.get("count", 0)
            
        except Exception as e:
            logger.error(f"❌ Erreur count: {str(e)}")
            raise ElasticsearchQueryError(f"Erreur lors du count: {str(e)}")
    
    async def get_mapping(self) -> Dict[str, Any]:
        """
        Récupère le mapping de l'index.
        
        Returns:
            Dict: Mapping de l'index
        """
        if not self.client:
            raise ElasticsearchQueryError("Client Elasticsearch non connecté")
        
        try:
            response = await self.client.indices.get_mapping(index=self.index_name)
            return response.get(self.index_name, {}).get("mappings", {})
            
        except Exception as e:
            logger.error(f"❌ Erreur get_mapping: {str(e)}")
            raise ElasticsearchQueryError(f"Erreur lors de la récupération du mapping: {str(e)}")
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Indexe plusieurs documents en batch.
        
        Args:
            documents: Liste de documents à indexer
            
        Returns:
            Dict: Résultat de l'indexation
        """
        if not self.client:
            raise ElasticsearchQueryError("Client Elasticsearch non connecté")
        
        try:
            # Préparation des documents pour bulk
            actions = []
            for doc in documents:
                action = {
                    "_index": self.index_name,
                    "_source": doc
                }
                
                # Utiliser transaction_id comme document ID si disponible
                if "transaction_id" in doc:
                    action["_id"] = doc["transaction_id"]
                
                actions.append(action)
            
            # Indexation en batch
            success_count, errors = await async_bulk(
                self.client,
                actions,
                chunk_size=1000,
                max_chunk_bytes=10 * 1024 * 1024  # 10MB par chunk
            )
            
            logger.info(f"✅ Indexation bulk: {success_count} documents, {len(errors)} erreurs")
            
            return {
                "success_count": success_count,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur bulk_index: {str(e)}")
            raise ElasticsearchQueryError(f"Erreur lors de l'indexation bulk: {str(e)}")
    
    async def refresh_index(self) -> None:
        """Force le rafraîchissement de l'index."""
        if not self.client:
            raise ElasticsearchQueryError("Client Elasticsearch non connecté")
        
        try:
            await self.client.indices.refresh(index=self.index_name)
            logger.debug("✅ Index rafraîchi")
            
        except Exception as e:
            logger.error(f"❌ Erreur refresh_index: {str(e)}")
            raise ElasticsearchQueryError(f"Erreur lors du rafraîchissement: {str(e)}")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de l'index.
        
        Returns:
            Dict: Statistiques de l'index
        """
        if not self.client:
            raise ElasticsearchQueryError("Client Elasticsearch non connecté")
        
        try:
            response = await self.client.indices.stats(index=self.index_name)
            return response.get("indices", {}).get(self.index_name, {})
            
        except Exception as e:
            logger.error(f"❌ Erreur get_index_stats: {str(e)}")
            raise ElasticsearchQueryError(f"Erreur lors de la récupération des stats: {str(e)}")
    
    def get_client_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du client.
        
        Returns:
            Dict: Métriques de performance
        """
        avg_query_time = (
            self.total_query_time / self.query_count 
            if self.query_count > 0 else 0.0
        )
        
        return {
            "query_count": self.query_count,
            "total_query_time_ms": self.total_query_time,
            "average_query_time_ms": avg_query_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.query_count, 1),
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "circuit_breaker_open": self.circuit_breaker_open,
            "circuit_breaker_failures": self.circuit_breaker_failures
        }
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()


# === HELPER FUNCTIONS ===

def build_financial_query(
    user_id: int,
    text_query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    date_range: Optional[Dict[str, str]] = None,
    amount_range: Optional[Dict[str, float]] = None,
    categories: Optional[List[str]] = None,
    merchants: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Construit une requête Elasticsearch optimisée pour les données financières.
    
    Args:
        user_id: ID de l'utilisateur (obligatoire)
        text_query: Recherche textuelle
        filters: Filtres additionnels
        date_range: Plage de dates (from, to)
        amount_range: Plage de montants (min, max)
        categories: Liste des catégories
        merchants: Liste des marchands
        
    Returns:
        Dict: Requête Elasticsearch optimisée
    """
    # Clause obligatoire : user_id
    must_clauses = [
        {"term": {"user_id": user_id}}
    ]
    
    # Recherche textuelle avec boost
    if text_query:
        text_query = text_query.strip()
        if text_query:
            must_clauses.append({
                "multi_match": {
                    "query": text_query,
                    "fields": [
                        "searchable_text^2.0",
                        "primary_description^1.5",
                        "merchant_name^1.8",
                        "category_name^1.2"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "operator": "and"
                }
            })
    
    # Filtres de plage
    filter_clauses = []
    
    # Plage de dates
    if date_range:
        if date_range.get("from") or date_range.get("to"):
            range_filter = {"date": {}}
            if date_range.get("from"):
                range_filter["date"]["gte"] = date_range["from"]
            if date_range.get("to"):
                range_filter["date"]["lte"] = date_range["to"]
            filter_clauses.append({"range": range_filter})
    
    # Plage de montants
    if amount_range:
        if amount_range.get("min") is not None or amount_range.get("max") is not None:
            range_filter = {"amount_abs": {}}
            if amount_range.get("min") is not None:
                range_filter["amount_abs"]["gte"] = amount_range["min"]
            if amount_range.get("max") is not None:
                range_filter["amount_abs"]["lte"] = amount_range["max"]
            filter_clauses.append({"range": range_filter})
    
    # Filtres catégories
    if categories:
        filter_clauses.append({
            "terms": {"category_name.keyword": categories}
        })
    
    # Filtres marchands
    if merchants:
        filter_clauses.append({
            "terms": {"merchant_name.keyword": merchants}
        })
    
    # Filtres additionnels
    if filters:
        for field, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({"terms": {f"{field}.keyword": value}})
            else:
                filter_clauses.append({"term": {f"{field}.keyword": value}})
    
    # Construction de la requête finale
    query = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        },
        "sort": [
            {"_score": {"order": "desc"}},
            {"date": {"order": "desc"}},
            {"amount_abs": {"order": "desc"}}
        ]
    }
    
    # Ajout des filtres si présents
    if filter_clauses:
        query["query"]["bool"]["filter"] = filter_clauses
    
    return query


def build_aggregation_query(
    user_id: int,
    base_query: Optional[Dict[str, Any]] = None,
    group_by_month: bool = False,
    group_by_category: bool = False,
    group_by_merchant: bool = False,
    include_stats: bool = True
) -> Dict[str, Any]:
    """
    Construit une requête d'agrégation pour les données financières.
    
    Args:
        user_id: ID de l'utilisateur
        base_query: Requête de base pour filtrer
        group_by_month: Grouper par mois
        group_by_category: Grouper par catégorie
        group_by_merchant: Grouper par marchand
        include_stats: Inclure les statistiques
        
    Returns:
        Dict: Requête d'agrégation Elasticsearch
    """
    # Requête de base ou filtrage par user_id
    if base_query:
        query = base_query.copy()
    else:
        query = {
            "query": {
                "bool": {
                    "must": [{"term": {"user_id": user_id}}]
                }
            }
        }
    
    # Ne retourner aucun document, juste les agrégations
    query["size"] = 0
    
    # Construction des agrégations
    aggs = {}
    
    # Statistiques globales
    if include_stats:
        aggs["total_amount"] = {
            "sum": {"field": "amount"}
        }
        aggs["total_amount_abs"] = {
            "sum": {"field": "amount_abs"}
        }
        aggs["avg_amount"] = {
            "avg": {"field": "amount_abs"}
        }
        aggs["min_amount"] = {
            "min": {"field": "amount_abs"}
        }
        aggs["max_amount"] = {
            "max": {"field": "amount_abs"}
        }
        aggs["amount_stats"] = {
            "stats": {"field": "amount_abs"}
        }
    
    # Agrégation par mois
    if group_by_month:
        aggs["by_month"] = {
            "terms": {
                "field": "month_year",
                "size": 24,  # 2 ans de données max
                "order": {"_key": "desc"}
            },
            "aggs": {
                "total_amount": {"sum": {"field": "amount"}},
                "total_amount_abs": {"sum": {"field": "amount_abs"}},
                "avg_amount": {"avg": {"field": "amount_abs"}}
            }
        }
    
    # Agrégation par catégorie
    if group_by_category:
        aggs["by_category"] = {
            "terms": {
                "field": "category_name.keyword",
                "size": 50,
                "order": {"total_amount_abs": "desc"}
            },
            "aggs": {
                "total_amount": {"sum": {"field": "amount"}},
                "total_amount_abs": {"sum": {"field": "amount_abs"}},
                "avg_amount": {"avg": {"field": "amount_abs"}}
            }
        }
    
    # Agrégation par marchand
    if group_by_merchant:
        aggs["by_merchant"] = {
            "terms": {
                "field": "merchant_name.keyword",
                "size": 30,
                "order": {"total_amount_abs": "desc"}
            },
            "aggs": {
                "total_amount": {"sum": {"field": "amount"}},
                "total_amount_abs": {"sum": {"field": "amount_abs"}},
                "avg_amount": {"avg": {"field": "amount_abs"}}
            }
        }
    
    # Ajout des agrégations à la requête
    if aggs:
        query["aggs"] = aggs
    
    return query


def optimize_query_for_performance(query: Dict[str, Any], user_id: int) -> Dict[str, Any]:
    """
    Optimise une requête Elasticsearch pour les performances.
    
    Args:
        query: Requête à optimiser
        user_id: ID de l'utilisateur pour le routage
        
    Returns:
        Dict: Requête optimisée
    """
    optimized = query.copy()
    
    # Préférence de routage basée sur l'user_id
    # Aide à diriger les requêtes vers les shards appropriés
    optimized["preference"] = f"_shards:{user_id % 5}"
    
    # Cache des résultats pour les requêtes répétées
    optimized["request_cache"] = True
    
    # Limitation du score tracking si pas nécessaire
    if "sort" in optimized and optimized["sort"]:
        optimized["track_scores"] = False
    
    # Optimisation des sources retournées
    if "_source" not in optimized:
        optimized["_source"] = [
            "transaction_id", "user_id", "account_id",
            "amount", "amount_abs", "transaction_type", "currency_code",
            "date", "month_year", "weekday",
            "primary_description", "merchant_name", "category_name", "operation_type"
        ]
    
    return optimized