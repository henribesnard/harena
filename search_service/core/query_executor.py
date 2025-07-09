"""
Exécuteur de requêtes Elasticsearch haute performance.

Ce module gère l'exécution optimisée des requêtes Elasticsearch
avec parallélisation, retry et gestion d'erreurs avancée.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import json

from ..clients.elasticsearch_client import ElasticsearchClient, ElasticsearchQueryError
from ..config.settings import SearchServiceSettings, get_settings
from ..utils.metrics import QueryExecutionMetrics


logger = logging.getLogger(__name__)


class QueryExecutionStrategy(str, Enum):
    """Stratégies d'exécution de requêtes."""
    SINGLE = "single"                 # Requête unique
    PARALLEL = "parallel"             # Requêtes parallèles
    SEQUENTIAL = "sequential"         # Requêtes séquentielles
    BATCH = "batch"                   # Traitement par batch


class QueryPriority(str, Enum):
    """Priorités d'exécution."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class QueryExecutionResult:
    """Résultat d'exécution d'une requête."""
    
    def __init__(
        self,
        query_id: str,
        response: Dict[str, Any],
        execution_time_ms: int,
        elasticsearch_took: int,
        strategy_used: QueryExecutionStrategy,
        optimizations_applied: List[str],
        cache_hit: bool = False,
        error: Optional[Exception] = None
    ):
        self.query_id = query_id
        self.response = response
        self.execution_time_ms = execution_time_ms
        self.elasticsearch_took = elasticsearch_took
        self.strategy_used = strategy_used
        self.optimizations_applied = optimizations_applied
        self.cache_hit = cache_hit
        self.error = error
        self.success = error is None
        self.timestamp = datetime.utcnow()


class QueryExecutor:
    """
    Exécuteur de requêtes Elasticsearch haute performance.
    
    Fonctionnalités:
    - Exécution de requêtes simples et complexes
    - Parallélisation automatique des requêtes multiples
    - Gestion des priorités et de la charge
    - Optimisations de performance dynamiques
    - Retry automatique avec backoff
    - Métriques détaillées d'exécution
    """
    
    def __init__(
        self,
        elasticsearch_client: ElasticsearchClient,
        settings: Optional[SearchServiceSettings] = None
    ):
        self.elasticsearch_client = elasticsearch_client
        self.settings = settings or get_settings()
        
        # Métriques d'exécution
        self.metrics = QueryExecutionMetrics()
        
        # Pool de tâches pour parallélisation
        self.max_concurrent_queries = self.settings.MAX_CONCURRENT_REQUESTS
        self.semaphore = asyncio.Semaphore(self.max_concurrent_queries)
        
        # Configuration retry
        self.max_retries = 3
        self.base_retry_delay = 1.0
        self.max_retry_delay = 10.0
        
        # Compteurs de performance
        self.queries_executed = 0
        self.total_execution_time = 0.0
        self.parallel_queries_count = 0
        self.retry_count = 0
        self.failed_queries_count = 0
        
        logger.info("Query executor initialized with optimized configuration")
    
    async def execute_single_query(
        self,
        query_id: str,
        elasticsearch_query: Dict[str, Any],
        size: int = 20,
        from_: int = 0,
        timeout_ms: int = 5000,
        priority: QueryPriority = QueryPriority.NORMAL,
        user_id: Optional[int] = None
    ) -> QueryExecutionResult:
        """
        Exécute une requête Elasticsearch unique.
        
        Args:
            query_id: Identifiant unique de la requête
            elasticsearch_query: Requête Elasticsearch
            size: Nombre de résultats
            from_: Offset pour pagination
            timeout_ms: Timeout en millisecondes
            priority: Priorité d'exécution
            user_id: ID utilisateur pour optimisations
            
        Returns:
            QueryExecutionResult: Résultat d'exécution
        """
        start_time = datetime.utcnow()
        optimizations = []
        
        try:
            async with self.semaphore:  # Limitation de concurrence
                # Application d'optimisations basées sur la priorité
                optimized_query = self._apply_query_optimizations(
                    elasticsearch_query, priority, user_id
                )
                optimizations.extend(self._get_applied_optimizations(optimized_query, elasticsearch_query))
                
                # Calcul du timeout adaptatif
                adaptive_timeout = self._calculate_adaptive_timeout(timeout_ms, priority)
                
                # Exécution avec retry
                response = await self._execute_with_retry(
                    query_id=query_id,
                    query=optimized_query,
                    size=size,
                    from_=from_,
                    timeout=f"{adaptive_timeout}ms",
                    preference=self._get_routing_preference(user_id)
                )
                
                # Calcul du temps d'exécution
                execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                # Mise à jour des métriques
                self._update_execution_metrics(execution_time_ms, priority, True)
                
                result = QueryExecutionResult(
                    query_id=query_id,
                    response=response,
                    execution_time_ms=execution_time_ms,
                    elasticsearch_took=response.get("took", 0),
                    strategy_used=QueryExecutionStrategy.SINGLE,
                    optimizations_applied=optimizations
                )
                
                logger.debug(f"✅ Requête {query_id} exécutée en {execution_time_ms}ms")
                return result
                
        except Exception as e:
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._update_execution_metrics(execution_time_ms, priority, False)
            
            logger.error(f"❌ Erreur exécution requête {query_id}: {str(e)}")
            
            return QueryExecutionResult(
                query_id=query_id,
                response={},
                execution_time_ms=execution_time_ms,
                elasticsearch_took=0,
                strategy_used=QueryExecutionStrategy.SINGLE,
                optimizations_applied=optimizations,
                error=e
            )
    
    async def execute_parallel_queries(
        self,
        queries: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
        priority: QueryPriority = QueryPriority.NORMAL
    ) -> List[QueryExecutionResult]:
        """
        Exécute plusieurs requêtes en parallèle.
        
        Args:
            queries: Liste de tuples (query_id, elasticsearch_query, params)
            priority: Priorité globale d'exécution
            
        Returns:
            List[QueryExecutionResult]: Résultats d'exécution
        """
        start_time = datetime.utcnow()
        
        logger.info(f"🚀 Exécution parallèle de {len(queries)} requêtes")
        
        # Création des tâches asynchrones
        tasks = []
        for query_id, elasticsearch_query, params in queries:
            task = asyncio.create_task(
                self.execute_single_query(
                    query_id=query_id,
                    elasticsearch_query=elasticsearch_query,
                    size=params.get("size", 20),
                    from_=params.get("from", 0),
                    timeout_ms=params.get("timeout_ms", 5000),
                    priority=priority,
                    user_id=params.get("user_id")
                )
            )
            tasks.append(task)
        
        # Exécution en parallèle avec gestion des erreurs
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Traitement des résultats et exceptions
            processed_results = []
            success_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Création d'un résultat d'erreur
                    query_id, _, _ = queries[i]
                    error_result = QueryExecutionResult(
                        query_id=query_id,
                        response={},
                        execution_time_ms=0,
                        elasticsearch_took=0,
                        strategy_used=QueryExecutionStrategy.PARALLEL,
                        optimizations_applied=[],
                        error=result
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
                    if result.success:
                        success_count += 1
            
            # Mise à jour des statistiques
            total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.parallel_queries_count += 1
            
            logger.info(
                f"✅ Exécution parallèle terminée: {success_count}/{len(queries)} succès en {total_time}ms"
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution parallèle: {str(e)}")
            raise ElasticsearchQueryError(f"Échec de l'exécution parallèle: {str(e)}")
    
    async def execute_batch_queries(
        self,
        queries: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
        batch_size: int = 5,
        priority: QueryPriority = QueryPriority.NORMAL
    ) -> List[QueryExecutionResult]:
        """
        Exécute des requêtes par batches pour gérer la charge.
        
        Args:
            queries: Liste de requêtes à exécuter
            batch_size: Taille des batches
            priority: Priorité d'exécution
            
        Returns:
            List[QueryExecutionResult]: Résultats agrégés
        """
        logger.info(f"📦 Exécution par batches: {len(queries)} requêtes, taille batch: {batch_size}")
        
        all_results = []
        
        # Division en batches
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_number = (i // batch_size) + 1
            
            logger.debug(f"🔄 Traitement batch {batch_number}: {len(batch)} requêtes")
            
            # Exécution du batch
            batch_results = await self.execute_parallel_queries(batch, priority)
            all_results.extend(batch_results)
            
            # Pause entre batches si nécessaire
            if i + batch_size < len(queries):
                await asyncio.sleep(0.1)  # Pause courte pour éviter la surcharge
        
        logger.info(f"✅ Exécution par batches terminée: {len(all_results)} résultats")
        return all_results
    
    async def execute_aggregation_query(
        self,
        query_id: str,
        base_query: Dict[str, Any],
        aggregations: Dict[str, Any],
        timeout_ms: int = 10000,
        user_id: Optional[int] = None
    ) -> QueryExecutionResult:
        """
        Exécute une requête d'agrégation optimisée.
        
        Args:
            query_id: Identifiant de la requête
            base_query: Requête de base pour filtrage
            aggregations: Configuration des agrégations
            timeout_ms: Timeout spécifique aux agrégations
            user_id: ID utilisateur
            
        Returns:
            QueryExecutionResult: Résultat avec agrégations
        """
        # Construction de la requête d'agrégation
        aggregation_query = {
            **base_query,
            "size": 0,  # Pas besoin de documents pour les agrégations
            "aggs": aggregations,
            "timeout": f"{timeout_ms}ms"
        }
        
        # Optimisations spécifiques aux agrégations
        optimized_query = self._optimize_aggregation_query(aggregation_query, user_id)
        
        # Exécution avec priorité élevée (agrégations sont souvent critiques)
        return await self.execute_single_query(
            query_id=query_id,
            elasticsearch_query=optimized_query,
            size=0,
            timeout_ms=timeout_ms,
            priority=QueryPriority.HIGH,
            user_id=user_id
        )
    
    async def _execute_with_retry(
        self,
        query_id: str,
        query: Dict[str, Any],
        size: int,
        from_: int,
        timeout: str,
        preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Exécute une requête avec retry automatique.
        
        Args:
            query_id: ID de la requête
            query: Requête Elasticsearch
            size: Taille des résultats
            from_: Offset
            timeout: Timeout
            preference: Préférence de routage
            
        Returns:
            Dict: Réponse Elasticsearch
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.elasticsearch_client.search(
                    query=query,
                    size=size,
                    from_=from_,
                    timeout=timeout,
                    preference=preference
                )
                
                if attempt > 0:
                    logger.info(f"✅ Requête {query_id} réussie après {attempt} tentatives")
                
                return response
                
            except Exception as e:
                last_exception = e
                self.retry_count += 1
                
                if attempt < self.max_retries:
                    # Calcul du délai avec backoff exponentiel
                    delay = min(
                        self.base_retry_delay * (2 ** attempt),
                        self.max_retry_delay
                    )
                    
                    logger.warning(
                        f"⚠️ Tentative {attempt + 1}/{self.max_retries + 1} échouée pour {query_id}: {str(e)}. "
                        f"Retry dans {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ Toutes les tentatives échouées pour {query_id}")
        
        self.failed_queries_count += 1
        raise last_exception
    
    def _apply_query_optimizations(
        self,
        query: Dict[str, Any],
        priority: QueryPriority,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Applique des optimisations à une requête selon la priorité.
        
        Args:
            query: Requête originale
            priority: Priorité d'exécution
            user_id: ID utilisateur pour optimisations
            
        Returns:
            Dict: Requête optimisée
        """
        optimized = query.copy()
        
        # Cache selon la priorité
        if priority in [QueryPriority.LOW, QueryPriority.NORMAL]:
            optimized["request_cache"] = True
        
        # Optimisation des sources selon la priorité
        if priority == QueryPriority.LOW and "_source" not in optimized:
            # Sources limitées pour les requêtes de faible priorité
            optimized["_source"] = [
                "transaction_id", "user_id", "amount", "date", "primary_description"
            ]
        
        # Limitation du scoring pour les requêtes de tri
        if "sort" in optimized and optimized["sort"]:
            optimized["track_scores"] = False
        
        # Préférence de routage basée sur user_id
        if user_id and "preference" not in optimized:
            optimized["preference"] = f"_shards:{user_id % 5}"
        
        return optimized
    
    def _optimize_aggregation_query(
        self,
        query: Dict[str, Any],
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimise une requête d'agrégation.
        
        Args:
            query: Requête d'agrégation
            user_id: ID utilisateur
            
        Returns:
            Dict: Requête optimisée pour agrégations
        """
        optimized = query.copy()
        
        # Désactivation du cache pour les agrégations (souvent uniques)
        optimized["request_cache"] = False
        
        # Optimisation des buckets selon la configuration
        if "aggs" in optimized:
            for agg_name, agg_config in optimized["aggs"].items():
                if isinstance(agg_config, dict) and "terms" in agg_config:
                    # Limitation du nombre de buckets
                    if "size" not in agg_config["terms"]:
                        agg_config["terms"]["size"] = min(50, self.settings.MAX_AGGREGATION_BUCKETS)
        
        # Timeout spécifique aux agrégations
        if "timeout" not in optimized:
            optimized["timeout"] = f"{self.settings.AGGREGATION_TIMEOUT_MS}ms"
        
        return optimized
    
    def _calculate_adaptive_timeout(
        self,
        base_timeout_ms: int,
        priority: QueryPriority
    ) -> int:
        """
        Calcule un timeout adaptatif selon la priorité.
        
        Args:
            base_timeout_ms: Timeout de base
            priority: Priorité de la requête
            
        Returns:
            int: Timeout adapté
        """
        multipliers = {
            QueryPriority.LOW: 0.8,
            QueryPriority.NORMAL: 1.0,
            QueryPriority.HIGH: 1.5,
            QueryPriority.CRITICAL: 2.0
        }
        
        multiplier = multipliers.get(priority, 1.0)
        adapted_timeout = int(base_timeout_ms * multiplier)
        
        # Respect des limites configurées
        return min(adapted_timeout, self.settings.SEARCH_MAX_TIMEOUT_MS)
    
    def _get_routing_preference(self, user_id: Optional[int]) -> Optional[str]:
        """
        Calcule la préférence de routage pour optimiser les performances.
        
        Args:
            user_id: ID utilisateur
            
        Returns:
            Optional[str]: Préférence de routage
        """
        if user_id:
            # Routage basé sur l'user_id pour localité des données
            return f"_shards:{user_id % 5}"
        return None
    
    def _get_applied_optimizations(
        self,
        optimized_query: Dict[str, Any],
        original_query: Dict[str, Any]
    ) -> List[str]:
        """
        Détermine les optimisations appliquées à une requête.
        
        Args:
            optimized_query: Requête optimisée
            original_query: Requête originale
            
        Returns:
            List[str]: Liste des optimisations appliquées
        """
        optimizations = []
        
        if optimized_query.get("request_cache") and not original_query.get("request_cache"):
            optimizations.append("request_caching")
        
        if optimized_query.get("track_scores") == False and original_query.get("track_scores") != False:
            optimizations.append("score_tracking_disabled")
        
        if optimized_query.get("preference") and not original_query.get("preference"):
            optimizations.append("shard_routing")
        
        if optimized_query.get("_source") != original_query.get("_source"):
            optimizations.append("source_filtering")
        
        return optimizations
    
    def _update_execution_metrics(
        self,
        execution_time_ms: int,
        priority: QueryPriority,
        success: bool
    ) -> None:
        """
        Met à jour les métriques d'exécution.
        
        Args:
            execution_time_ms: Temps d'exécution
            priority: Priorité de la requête
            success: Succès de l'exécution
        """
        self.queries_executed += 1
        self.total_execution_time += execution_time_ms
        
        if not success:
            self.failed_queries_count += 1
        
        # Mise à jour des métriques détaillées
        self.metrics.record_execution(
            execution_time_ms=execution_time_ms,
            priority=priority.value,
            success=success
        )
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques d'exécution.
        
        Returns:
            Dict: Métriques détaillées
        """
        avg_execution_time = (
            self.total_execution_time / self.queries_executed
            if self.queries_executed > 0 else 0.0
        )
        
        success_rate = (
            (self.queries_executed - self.failed_queries_count) / self.queries_executed
            if self.queries_executed > 0 else 0.0
        )
        
        return {
            "queries_executed": self.queries_executed,
            "total_execution_time_ms": self.total_execution_time,
            "average_execution_time_ms": avg_execution_time,
            "parallel_queries_count": self.parallel_queries_count,
            "retry_count": self.retry_count,
            "failed_queries_count": self.failed_queries_count,
            "success_rate": success_rate,
            "max_concurrent_queries": self.max_concurrent_queries,
            "detailed_metrics": self.metrics.get_metrics()
        }
    
    def reset_metrics(self) -> None:
        """Remet à zéro les métriques d'exécution."""
        self.queries_executed = 0
        self.total_execution_time = 0.0
        self.parallel_queries_count = 0
        self.retry_count = 0
        self.failed_queries_count = 0
        self.metrics.reset()
        
        logger.info("🔄 Métriques d'exécution réinitialisées")
    
    async def shutdown(self) -> None:
        """Arrête proprement l'exécuteur de requêtes."""
        logger.info("🛑 Arrêt de l'exécuteur de requêtes...")
        
        # Attendre que toutes les tâches en cours se terminent
        pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if pending_tasks:
            logger.info(f"⏳ Attente de {len(pending_tasks)} tâches en cours...")
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        
        # Logs des métriques finales
        final_metrics = self.get_execution_metrics()
        logger.info(f"📊 Métriques finales: {final_metrics}")
        
        logger.info("✅ Arrêt terminé")


# === HELPER FUNCTIONS ===

def create_query_executor(
    elasticsearch_client: ElasticsearchClient,
    settings: Optional[SearchServiceSettings] = None
) -> QueryExecutor:
    """
    Factory pour créer un exécuteur de requêtes.
    
    Args:
        elasticsearch_client: Client Elasticsearch
        settings: Configuration
        
    Returns:
        QueryExecutor: Exécuteur configuré
    """
    return QueryExecutor(
        elasticsearch_client=elasticsearch_client,
        settings=settings or get_settings()
    )