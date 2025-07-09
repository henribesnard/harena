"""
Lexical Engine - Orchestrateur principal du moteur de recherche

Responsabilité : Point d'entrée principal qui orchestre tous les composants core
pour fournir une interface de recherche lexicale haute performance et optimisée.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from elasticsearch import AsyncElasticsearch

from .query_executor import QueryExecutor
from .result_processor import ResultProcessor
from .performance_optimizer import PerformanceOptimizer
from ..models.service_contracts import SearchServiceQuery, SearchServiceResponse
from ..clients.elasticsearch_client import ElasticsearchClient
from ..utils.metrics import MetricsCollector
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class LexicalEngine:
    """
    Moteur de recherche lexicale principal - Orchestrateur haute performance.
    
    Responsabilités:
    - Orchestration complète du pipeline de recherche
    - Intégration Query Executor + Result Processor + Performance Optimizer
    - Interface unifiée pour l'API search service
    - Gestion des erreurs et résilience
    - Monitoring et métriques centralisées
    - Support multi-search et batch processing
    """
    
    def __init__(
        self, 
        elasticsearch_client: Optional[AsyncElasticsearch] = None,
        enable_optimization: bool = True,
        cache_size: int = 1000
    ):
        self.settings = get_settings()
        
        # Initialisation des composants core
        self.es_client = elasticsearch_client or self._create_elasticsearch_client()
        self.query_executor = QueryExecutor(self.es_client)
        self.result_processor = ResultProcessor()
        
        # Optimisation optionnelle (peut être désactivée pour debug)
        self.enable_optimization = enable_optimization
        if enable_optimization:
            self.performance_optimizer = PerformanceOptimizer(cache_size)
        else:
            self.performance_optimizer = None
        
        # Monitoring et métriques
        self.metrics_collector = MetricsCollector()
        
        # Statistiques globales du moteur
        self.engine_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_processing_time": 0.0,
            "avg_search_time": 0.0,
            "uptime_start": datetime.now()
        }
        
        logger.info(
            "LexicalEngine initialisé (optimisation: %s, cache: %d)",
            enable_optimization, cache_size if enable_optimization else 0
        )
    
    async def search(self, query_contract: SearchServiceQuery) -> SearchServiceResponse:
        """
        Point d'entrée principal pour une recherche unique.
        
        Args:
            query_contract: Contrat de requête standardisé
            
        Returns:
            SearchServiceResponse: Réponse complète traitée et optimisée
            
        Raises:
            SearchEngineException: Erreur lors de la recherche
            ValidationException: Contrat invalide
        """
        search_start_time = datetime.now()
        
        try:
            # 1. Validation du contrat
            self._validate_search_contract(query_contract)
            
            # 2. Execution avec ou sans optimisation
            if self.enable_optimization and self.performance_optimizer:
                response, perf_metadata = await self.performance_optimizer.optimize_and_cache_search(
                    query_contract, 
                    self._execute_search_internal
                )
            else:
                response = await self._execute_search_internal(query_contract)
                perf_metadata = {"optimization_enabled": False}
            
            # 3. Enrichissement des métadonnées de performance
            search_time = (datetime.now() - search_start_time).total_seconds() * 1000
            response.performance.update(perf_metadata)
            response.performance["total_engine_time_ms"] = search_time
            
            # 4. Collecte des métriques
            await self._collect_search_metrics(query_contract, response, search_time, True)
            
            # 5. Mise à jour des statistiques
            self._update_engine_stats(search_time, True)
            
            logger.info(
                f"Recherche réussie: query_id={query_contract.query_metadata.query_id}, "
                f"résultats={response.response_metadata.total_hits}, "
                f"temps={search_time:.2f}ms"
            )
            
            return response
            
        except Exception as e:
            # Gestion des erreurs avec logging détaillé
            search_time = (datetime.now() - search_start_time).total_seconds() * 1000
            await self._handle_search_error(query_contract, e, search_time)
            self._update_engine_stats(search_time, False)
            raise
    
    async def multi_search(
        self, 
        query_contracts: List[SearchServiceQuery],
        max_parallel: int = 5
    ) -> List[SearchServiceResponse]:
        """
        Exécute plusieurs recherches en parallèle avec optimisations batch.
        
        Args:
            query_contracts: Liste des contrats de requête
            max_parallel: Nombre maximum de requêtes parallèles
            
        Returns:
            List[SearchServiceResponse]: Liste des réponses
        """
        if not query_contracts:
            return []
        
        multi_search_start = datetime.now()
        
        try:
            # 1. Validation de tous les contrats
            for contract in query_contracts:
                self._validate_search_contract(contract)
            
            # 2. Exécution optimisée ou standard
            if self.enable_optimization and self.performance_optimizer:
                # Utilisation du batch optimizer
                results = await self.performance_optimizer.batch_optimize_searches(
                    query_contracts, 
                    self._execute_search_internal,
                    max_parallel
                )
                
                # Extraction des réponses depuis les tuples (response, metadata)
                responses = []
                for result in results:
                    if isinstance(result, tuple) and len(result) == 2:
                        response, perf_metadata = result
                        if response:  # Pas d'erreur
                            response.performance.update(perf_metadata)
                            responses.append(response)
                        else:  # Erreur dans le batch
                            error_response = self._create_error_response(
                                query_contracts[len(responses)], 
                                Exception(perf_metadata.get("error", "Erreur inconnue"))
                            )
                            responses.append(error_response)
                    else:
                        # Résultat inattendu
                        error_response = self._create_error_response(
                            query_contracts[len(responses)],
                            Exception("Format de résultat inattendu")
                        )
                        responses.append(error_response)
            
            else:
                # Exécution standard en parallèle
                semaphore = asyncio.Semaphore(max_parallel)
                
                async def execute_single(contract):
                    async with semaphore:
                        return await self._execute_search_internal(contract)
                
                responses = await asyncio.gather(
                    *[execute_single(contract) for contract in query_contracts],
                    return_exceptions=True
                )
                
                # Traitement des exceptions
                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        responses[i] = self._create_error_response(
                            query_contracts[i], response
                        )
            
            # 3. Collecte des métriques multi-search
            total_time = (datetime.now() - multi_search_start).total_seconds() * 1000
            await self._collect_multi_search_metrics(
                query_contracts, responses, total_time
            )
            
            logger.info(
                f"Multi-search complété: {len(query_contracts)} requêtes, "
                f"{len([r for r in responses if not hasattr(r, 'error')])} succès, "
                f"temps total: {total_time:.2f}ms"
            )
            
            return responses
            
        except Exception as e:
            logger.error(f"Erreur lors du multi-search: {str(e)}")
            # Retourner des réponses d'erreur pour toutes les requêtes
            return [
                self._create_error_response(contract, e) 
                for contract in query_contracts
            ]
    
    async def validate_search_syntax(
        self, 
        query_contract: SearchServiceQuery
    ) -> Dict[str, Any]:
        """
        Valide la syntaxe d'une requête sans l'exécuter.
        
        Args:
            query_contract: Contrat de requête à valider
            
        Returns:
            Dict: Résultat de validation avec détails
        """
        try:
            # 1. Validation du contrat
            self._validate_search_contract(query_contract)
            
            # 2. Validation de la syntaxe Elasticsearch
            validation_result = await self.query_executor.validate_query_syntax(
                query_contract
            )
            
            # 3. Validation des optimisations possibles
            optimization_suggestions = []
            if self.enable_optimization and self.performance_optimizer:
                optimization_suggestions = await self._suggest_optimizations(
                    query_contract
                )
            
            return {
                "contract_valid": True,
                "elasticsearch_valid": validation_result.get("valid", False),
                "elasticsearch_explanation": validation_result.get("explanation", []),
                "optimization_suggestions": optimization_suggestions,
                "estimated_performance": self._estimate_query_performance(query_contract),
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {str(e)}")
            return {
                "contract_valid": False,
                "elasticsearch_valid": False,
                "validation_error": str(e),
                "validation_timestamp": datetime.now().isoformat()
            }
    
    async def _execute_search_internal(
        self, 
        query_contract: SearchServiceQuery
    ) -> SearchServiceResponse:
        """
        Exécution interne de la recherche (utilisée par l'optimiseur et direct).
        
        Args:
            query_contract: Contrat de requête validé
            
        Returns:
            SearchServiceResponse: Réponse traitée
        """
        execution_start = datetime.now()
        
        try:
            # 1. Exécution de la requête Elasticsearch
            es_response = await self.query_executor.execute_search(query_contract)
            execution_time = (datetime.now() - execution_start).total_seconds() * 1000
            
            # 2. Traitement des résultats
            processed_response = await self.result_processor.process_search_response(
                es_response, query_contract, int(execution_time)
            )
            
            return processed_response
            
        except Exception as e:
            execution_time = (datetime.now() - execution_start).total_seconds() * 1000
            logger.error(
                f"Erreur lors de l'exécution interne: {str(e)}, "
                f"query_id={query_contract.query_metadata.query_id}"
            )
            raise
    
    def _validate_search_contract(self, query_contract: SearchServiceQuery) -> None:
        """
        Valide un contrat de recherche avant exécution.
        
        Args:
            query_contract: Contrat à valider
            
        Raises:
            ValueError: Contrat invalide
        """
        # Validation des champs obligatoires
        if not query_contract.query_metadata:
            raise ValueError("query_metadata est obligatoire")
        
        if not query_contract.query_metadata.user_id:
            raise ValueError("user_id est obligatoire dans query_metadata")
        
        if not query_contract.search_parameters:
            raise ValueError("search_parameters est obligatoire")
        
        # Validation sécurité: user_id filter obligatoire
        user_filter_exists = any(
            f.field == "user_id" and f.value == query_contract.query_metadata.user_id
            for f in query_contract.filters.required
        )
        
        if not user_filter_exists:
            raise ValueError("Filtre user_id obligatoire pour la sécurité")
        
        # Validation des limites
        if query_contract.search_parameters.limit > self.settings.MAX_SEARCH_SIZE:
            raise ValueError(
                f"Limite trop élevée: {query_contract.search_parameters.limit} > "
                f"{self.settings.MAX_SEARCH_SIZE}"
            )
        
        if query_contract.search_parameters.offset > self.settings.MAX_SEARCH_OFFSET:
            raise ValueError(
                f"Offset trop élevé: {query_contract.search_parameters.offset} > "
                f"{self.settings.MAX_SEARCH_OFFSET}"
            )
        
        # Validation timeout
        max_timeout = self.settings.ELASTICSEARCH_MAX_TIMEOUT_MS or 10000
        if query_contract.search_parameters.timeout_ms and query_contract.search_parameters.timeout_ms > max_timeout:
            raise ValueError(f"Timeout trop élevé: {query_contract.search_parameters.timeout_ms} > {max_timeout}")
    
    async def _suggest_optimizations(
        self, 
        query_contract: SearchServiceQuery
    ) -> List[str]:
        """
        Suggère des optimisations pour une requête.
        
        Args:
            query_contract: Contrat de requête à analyser
            
        Returns:
            List[str]: Liste des suggestions d'optimisation
        """
        suggestions = []
        
        # Suggestions basées sur la structure de la requête
        if not query_contract.search_parameters.fields:
            suggestions.append("Spécifier les champs nécessaires pour réduire la taille de transfert")
        
        if query_contract.search_parameters.limit > 100:
            suggestions.append("Considérer une pagination avec limite plus petite")
        
        if (query_contract.aggregations and 
            query_contract.aggregations.enabled and 
            query_contract.search_parameters.limit > 10):
            suggestions.append("Réduire la limite de résultats pour les requêtes d'agrégation")
        
        # Suggestions basées sur les filtres
        if len(query_contract.filters.required) > 5:
            suggestions.append("Considérer la consolidation des filtres pour améliorer les performances")
        
        if query_contract.filters.text_search and not query_contract.filters.text_search.get("fields"):
            suggestions.append("Spécifier les champs de recherche textuelle pour de meilleures performances")
        
        # Suggestions basées sur l'intention
        intent = query_contract.query_metadata.intent_type
        if intent in ["COUNT_OPERATIONS", "TEMPORAL_ANALYSIS"]:
            if query_contract.search_parameters.limit > 0:
                suggestions.append("Utiliser limit=0 pour les requêtes d'agrégation pure")
        
        return suggestions
    
    def _estimate_query_performance(self, query_contract: SearchServiceQuery) -> Dict[str, Any]:
        """
        Estime les performances d'une requête.
        
        Args:
            query_contract: Contrat de requête à analyser
            
        Returns:
            Dict: Estimation de performance
        """
        # Facteurs de complexité
        complexity_score = 0
        
        # Complexité des filtres
        complexity_score += len(query_contract.filters.required) * 0.1
        complexity_score += len(query_contract.filters.optional) * 0.05
        complexity_score += len(query_contract.filters.ranges) * 0.15
        
        # Complexité de la recherche textuelle
        if query_contract.filters.text_search:
            complexity_score += 0.3
            if query_contract.filters.text_search.get("operator") == "fuzzy":
                complexity_score += 0.2
        
        # Complexité des agrégations
        if query_contract.aggregations and query_contract.aggregations.enabled:
            complexity_score += 0.4
            complexity_score += len(query_contract.aggregations.group_by or []) * 0.1
            complexity_score += len(query_contract.aggregations.metrics or []) * 0.05
        
        # Complexité de la pagination
        if query_contract.search_parameters.offset > 1000:
            complexity_score += 0.3
        
        if query_contract.search_parameters.limit > 100:
            complexity_score += 0.2
        
        # Classification de la performance
        if complexity_score <= 0.5:
            performance_class = "fast"
            estimated_time_ms = "< 50ms"
        elif complexity_score <= 1.0:
            performance_class = "medium"
            estimated_time_ms = "50-200ms"
        elif complexity_score <= 2.0:
            performance_class = "slow"
            estimated_time_ms = "200-1000ms"
        else:
            performance_class = "very_slow"
            estimated_time_ms = "> 1000ms"
        
        return {
            "complexity_score": round(complexity_score, 2),
            "performance_class": performance_class,
            "estimated_time": estimated_time_ms,
            "bottlenecks": self._identify_performance_bottlenecks(query_contract)
        }
    
    def _identify_performance_bottlenecks(
        self, 
        query_contract: SearchServiceQuery
    ) -> List[str]:
        """
        Identifie les goulots d'étranglement potentiels.
        """
        bottlenecks = []
        
        if query_contract.search_parameters.offset > 5000:
            bottlenecks.append("deep_pagination")
        
        if query_contract.search_parameters.limit > 200:
            bottlenecks.append("large_result_set")
        
        if query_contract.filters.text_search and query_contract.filters.text_search.get("operator") == "fuzzy":
            bottlenecks.append("fuzzy_search")
        
        if (query_contract.aggregations and 
            query_contract.aggregations.enabled and 
            len(query_contract.aggregations.group_by or []) > 2):
            bottlenecks.append("complex_aggregations")
        
        if len(query_contract.filters.required) > 8:
            bottlenecks.append("too_many_filters")
        
        return bottlenecks
    
    async def _collect_search_metrics(
        self,
        query_contract: SearchServiceQuery,
        response: SearchServiceResponse,
        execution_time_ms: float,
        success: bool
    ) -> None:
        """
        Collecte les métriques pour une recherche unique.
        """
        await self.metrics_collector.record_search_metrics({
            "query_id": query_contract.query_metadata.query_id,
            "user_id": query_contract.query_metadata.user_id,
            "intent_type": query_contract.query_metadata.intent_type,
            "query_type": query_contract.search_parameters.query_type,
            "execution_time_ms": execution_time_ms,
            "results_count": response.response_metadata.total_hits if success else 0,
            "cache_hit": response.response_metadata.cache_hit if success else False,
            "success": success,
            "has_aggregations": bool(query_contract.aggregations and query_contract.aggregations.enabled),
            "filter_count": len(query_contract.filters.required),
            "timestamp": datetime.now()
        })
    
    async def _collect_multi_search_metrics(
        self,
        query_contracts: List[SearchServiceQuery],
        responses: List[SearchServiceResponse],
        total_time_ms: float
    ) -> None:
        """
        Collecte les métriques pour un multi-search.
        """
        successful_searches = len([r for r in responses if not hasattr(r, 'error')])
        
        await self.metrics_collector.record_multi_search_metrics({
            "total_queries": len(query_contracts),
            "successful_queries": successful_searches,
            "failed_queries": len(query_contracts) - successful_searches,
            "total_execution_time_ms": total_time_ms,
            "avg_time_per_query_ms": total_time_ms / len(query_contracts),
            "timestamp": datetime.now()
        })
    
    async def _handle_search_error(
        self,
        query_contract: SearchServiceQuery,
        error: Exception,
        execution_time_ms: float
    ) -> None:
        """
        Gère les erreurs de recherche avec logging détaillé.
        """
        error_info = {
            "query_id": query_contract.query_metadata.query_id,
            "user_id": query_contract.query_metadata.user_id,
            "intent_type": query_contract.query_metadata.intent_type,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now()
        }
        
        logger.error(f"Erreur de recherche: {error_info}")
        
        # Collecte des métriques d'erreur
        await self.metrics_collector.record_error_metrics(error_info)
    
    def _create_error_response(
        self,
        query_contract: SearchServiceQuery,
        error: Exception
    ) -> SearchServiceResponse:
        """
        Crée une réponse d'erreur standardisée.
        """
        from ..models.service_contracts import ResponseMetadata
        
        return SearchServiceResponse(
            response_metadata=ResponseMetadata(
                query_id=query_contract.query_metadata.query_id,
                execution_time_ms=0,
                total_hits=0,
                returned_hits=0,
                has_more=False,
                cache_hit=False,
                elasticsearch_took=0,
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": datetime.now().isoformat()
                }
            ),
            results=[],
            aggregations=None,
            performance={"error": True, "error_type": type(error).__name__},
            context_enrichment={
                "search_intent_matched": False,
                "result_quality_score": 0.0,
                "suggested_followup_questions": [
                    "Vérifiez votre requête",
                    "Contactez le support si le problème persiste"
                ]
            }
        )
    
    def _update_engine_stats(self, execution_time_ms: float, success: bool) -> None:
        """
        Met à jour les statistiques globales du moteur.
        """
        self.engine_stats["total_searches"] += 1
        
        if success:
            self.engine_stats["successful_searches"] += 1
            self.engine_stats["total_processing_time"] += execution_time_ms
            
            # Calcul de la moyenne mobile
            self.engine_stats["avg_search_time"] = (
                self.engine_stats["total_processing_time"] / 
                self.engine_stats["successful_searches"]
            )
        else:
            self.engine_stats["failed_searches"] += 1
    
    def _create_elasticsearch_client(self) -> AsyncElasticsearch:
        """
        Crée un client Elasticsearch par défaut.
        """
        es_client_manager = ElasticsearchClient()
        return es_client_manager.get_client()
    
    # === Méthodes de monitoring et administration ===
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Retourne le statut global du moteur.
        """
        uptime = datetime.now() - self.engine_stats["uptime_start"]
        
        status = {
            "status": "healthy",
            "uptime_seconds": uptime.total_seconds(),
            "engine_stats": self.engine_stats.copy(),
            "optimization_enabled": self.enable_optimization,
            "elasticsearch_connected": True  # TODO: vérification réelle
        }
        
        # Ajout des métriques d'optimisation si disponibles
        if self.enable_optimization and self.performance_optimizer:
            status["performance_metrics"] = self.performance_optimizer.get_performance_metrics()
            status["cache_statistics"] = self.performance_optimizer.get_cache_statistics()
        
        # Calcul du success rate
        total_searches = self.engine_stats["total_searches"]
        if total_searches > 0:
            status["success_rate"] = (
                self.engine_stats["successful_searches"] / total_searches
            )
        else:
            status["success_rate"] = 1.0
        
        # Détermination du statut de santé
        if status["success_rate"] < 0.95:
            status["status"] = "degraded"
        elif status["success_rate"] < 0.8:
            status["status"] = "unhealthy"
        
        return status
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques détaillées du moteur.
        """
        metrics = {
            "engine_metrics": self.engine_stats.copy(),
            "query_executor_stats": self.query_executor.get_cache_stats() if hasattr(self.query_executor, 'get_cache_stats') else {},
            "result_processor_stats": self.result_processor.get_processing_stats() if hasattr(self.result_processor, 'get_processing_stats') else {}
        }
        
        if self.enable_optimization and self.performance_optimizer:
            metrics["optimization_metrics"] = self.performance_optimizer.get_performance_metrics()
            metrics["cache_metrics"] = self.performance_optimizer.get_cache_statistics()
        
        if hasattr(self.metrics_collector, 'get_aggregated_metrics'):
            metrics["business_metrics"] = self.metrics_collector.get_aggregated_metrics()
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Effectue un health check complet du moteur.
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Test Elasticsearch
        try:
            info = await self.es_client.info()
            health_status["components"]["elasticsearch"] = {
                "status": "healthy",
                "version": info.get("version", {}).get("number", "unknown"),
                "cluster_name": info.get("cluster_name", "unknown")
            }
        except Exception as e:
            health_status["components"]["elasticsearch"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        
        # Test des composants core
        health_status["components"]["query_executor"] = {"status": "healthy"}
        health_status["components"]["result_processor"] = {"status": "healthy"}
        
        if self.enable_optimization and self.performance_optimizer:
            perf_metrics = self.performance_optimizer.get_performance_metrics()
            circuit_breaker_open = perf_metrics.get("circuit_breaker_status", {}).get("is_open", False)
            
            health_status["components"]["performance_optimizer"] = {
                "status": "degraded" if circuit_breaker_open else "healthy",
                "circuit_breaker_open": circuit_breaker_open
            }
            
            if circuit_breaker_open:
                health_status["overall_status"] = "degraded"
        
        return health_status
    
    async def clear_all_caches(self) -> Dict[str, str]:
        """
        Vide tous les caches du moteur.
        """
        cleared_caches = []
        
        if hasattr(self.query_executor, 'clear_cache'):
            await self.query_executor.clear_cache()
            cleared_caches.append("query_executor_cache")
        
        if self.enable_optimization and self.performance_optimizer:
            self.performance_optimizer.clear_all_caches()
            cleared_caches.append("performance_optimizer_caches")
        
        return {
            "status": "success",
            "cleared_caches": cleared_caches,
            "timestamp": datetime.now().isoformat()
        }
    
    async def reset_metrics(self) -> Dict[str, str]:
        """
        Remet à zéro toutes les métriques du moteur.
        """
        # Reset des stats du moteur
        self.engine_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_processing_time": 0.0,
            "avg_search_time": 0.0,
            "uptime_start": datetime.now()
        }
        
        # Reset des métriques des composants
        if hasattr(self.result_processor, 'reset_stats'):
            self.result_processor.reset_stats()
        
        if self.enable_optimization and self.performance_optimizer:
            self.performance_optimizer.reset_performance_metrics()
        
        return {
            "status": "success",
            "message": "Toutes les métriques ont été réinitialisées",
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self) -> None:
        """
        Arrêt propre du moteur.
        """
        logger.info("Arrêt du LexicalEngine en cours...")
        
        try:
            # Fermeture du client Elasticsearch
            if self.es_client:
                await self.es_client.close()
            
            # Sauvegarde des métriques si nécessaire
            if hasattr(self.metrics_collector, 'flush_metrics'):
                await self.metrics_collector.flush_metrics()
            
            logger.info("LexicalEngine arrêté proprement")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt du LexicalEngine: {str(e)}")
            raise