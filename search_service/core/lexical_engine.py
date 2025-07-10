"""
Moteur de recherche lexicale - Composant Core #4
Orchestrateur principal qui utilise tous les autres composants
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from search_service.core.query_executor import QueryExecutor
from search_service.core.result_processor import ResultProcessor
from search_service.core.performance_optimizer import PerformanceOptimizer, OptimizationLevel
from search_service.models.service_contracts import SearchServiceQuery, SearchServiceResponse
from search_service.templates.query_templates import QueryTemplateManager
from search_service.clients.elasticsearch_client import ElasticsearchClient
from search_service.utils.validators import QueryValidator
from search_service.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class LexicalEngine:
    """
    Moteur de recherche lexicale principal
    
    Responsabilités:
    - Orchestration de tous les composants de recherche
    - Exécution des recherches lexicales optimisées
    - Gestion du cache et des performances
    - Interface unifiée pour le Search Service
    """
    
    def __init__(
        self,
        elasticsearch_client: ElasticsearchClient,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        enable_metrics: bool = True
    ):
        self.es_client = elasticsearch_client
        self.optimization_level = optimization_level
        self.enable_metrics = enable_metrics
        
        # Initialisation des composants
        self._initialize_components()
        
        # État du moteur
        self.is_initialized = False
        self.startup_time = None
        
        logger.info("✅ LexicalEngine créé")
    
    def _initialize_components(self):
        """
        Initialise tous les composants du moteur
        """
        try:
            # Template manager
            self.template_manager = QueryTemplateManager()
            
            # Query executor
            self.query_executor = QueryExecutor(
                elasticsearch_client=self.es_client.client,
                template_manager=self.template_manager
            )
            
            # Result processor
            self.result_processor = ResultProcessor()
            
            # Performance optimizer
            self.performance_optimizer = PerformanceOptimizer(
                optimization_level=self.optimization_level
            )
            
            # Validator
            self.validator = QueryValidator()
            
            # Metrics collector
            if self.enable_metrics:
                self.metrics_collector = MetricsCollector()
            else:
                self.metrics_collector = None
            
            logger.info("✅ Tous les composants du moteur initialisés")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des composants: {e}")
            raise
    
    async def initialize(self):
        """
        Initialise le moteur de manière asynchrone
        """
        try:
            start_time = datetime.now()
            
            # Vérification de la connexion Elasticsearch
            await self.es_client.health_check()
            
            # Initialisation des templates
            await self.template_manager.initialize()
            
            # Préchauffage du cache si activé
            if self.optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
                await self._warm_cache()
            
            self.is_initialized = True
            self.startup_time = datetime.now() - start_time
            
            logger.info(f"✅ LexicalEngine initialisé en {self.startup_time.total_seconds():.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du moteur: {e}")
            raise
    
    async def search(self, query: SearchServiceQuery) -> SearchServiceResponse:
        """
        Exécute une recherche lexicale
        
        Args:
            query: Requête de recherche
            
        Returns:
            Réponse de recherche formatée
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = datetime.now()
            
            # Validation de la requête
            self.validator.validate_search_query(query)
            
            # Optimisation de la requête
            optimized_query = await self.performance_optimizer.optimize_query(query)
            
            # Exécution avec cache
            raw_results = await self.performance_optimizer.execute_with_cache(
                optimized_query,
                self.query_executor.execute_query
            )
            
            # Traitement des résultats
            processed_results = await self.result_processor.process_search_results(
                raw_results,
                query_text=query.query,
                include_aggregations=bool(query.aggregations)
            )
            
            # Calcul des scores de pertinence
            final_results = self.result_processor.calculate_relevance_scores(processed_results)
            
            # Métriques
            execution_time = datetime.now() - start_time
            if self.metrics_collector:
                await self._record_search_metrics(query, final_results, execution_time)
            
            logger.info(f"✅ Recherche exécutée: {len(final_results.hits)} résultats en {execution_time.total_seconds():.3f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            raise
    
    async def multi_search(self, queries: List[SearchServiceQuery]) -> List[SearchServiceResponse]:
        """
        Exécute plusieurs recherches en parallèle
        
        Args:
            queries: Liste des requêtes
            
        Returns:
            Liste des réponses
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = datetime.now()
            
            # Validation des requêtes
            for query in queries:
                self.validator.validate_search_query(query)
            
            # Optimisation des requêtes
            optimized_queries = []
            for query in queries:
                optimized = await self.performance_optimizer.optimize_query(query)
                optimized_queries.append(optimized)
            
            # Exécution en batch optimisé
            raw_results = await self.performance_optimizer.execute_batch_optimized(
                optimized_queries,
                self.query_executor.execute_query
            )
            
            # Traitement des résultats en parallèle
            processing_tasks = []
            for i, raw_result in enumerate(raw_results):
                task = self.result_processor.process_search_results(
                    raw_result,
                    query_text=queries[i].query,
                    include_aggregations=bool(queries[i].aggregations)
                )
                processing_tasks.append(task)
            
            processed_results = await asyncio.gather(*processing_tasks)
            
            # Calcul des scores de pertinence
            final_results = []
            for result in processed_results:
                final_result = self.result_processor.calculate_relevance_scores(result)
                final_results.append(final_result)
            
            # Métriques
            execution_time = datetime.now() - start_time
            if self.metrics_collector:
                await self._record_multi_search_metrics(queries, final_results, execution_time)
            
            total_hits = sum(len(result.hits) for result in final_results)
            logger.info(f"✅ Multi-recherche exécutée: {len(queries)} requêtes, {total_hits} résultats en {execution_time.total_seconds():.3f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la multi-recherche: {e}")
            raise
    
    async def search_with_aggregations(self, query: SearchServiceQuery) -> SearchServiceResponse:
        """
        Exécute une recherche avec focus sur les agrégations
        
        Args:
            query: Requête avec agrégations
            
        Returns:
            Réponse avec agrégations détaillées
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = datetime.now()
            
            # Validation
            self.validator.validate_search_query(query)
            if not query.aggregations:
                raise ValueError("Aucune agrégation spécifiée")
            
            # Optimisation spécifique aux agrégations
            optimized_query = await self.performance_optimizer.optimize_query(query)
            
            # Exécution des agrégations
            raw_results = await self.performance_optimizer.execute_with_cache(
                optimized_query,
                self.query_executor.execute_aggregation
            )
            
            # Traitement spécialisé pour les agrégations
            processed_results = await self.result_processor.process_search_results(
                raw_results,
                query_text=query.query,
                include_aggregations=True
            )
            
            # Métriques
            execution_time = datetime.now() - start_time
            if self.metrics_collector:
                await self._record_aggregation_metrics(query, processed_results, execution_time)
            
            logger.info(f"✅ Recherche avec agrégations: {len(processed_results.aggregations)} agrégations en {execution_time.total_seconds():.3f}s")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche avec agrégations: {e}")
            raise
    
    async def suggest_queries(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Suggère des requêtes basées sur un texte partiel
        
        Args:
            partial_query: Texte partiel
            limit: Nombre de suggestions
            
        Returns:
            Liste de suggestions
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Construction d'une requête de suggestion
            suggestion_query = SearchServiceQuery(
                query=partial_query,
                search_type="suggest",
                size=limit,
                index="financial_documents"
            )
            
            # Utilisation du template de suggestion
            suggestions = await self.template_manager.get_query_suggestions(partial_query, limit)
            
            logger.info(f"✅ {len(suggestions)} suggestions générées pour '{partial_query}'")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de suggestions: {e}")
            return []
    
    async def explain_query(self, query: SearchServiceQuery) -> Dict[str, Any]:
        """
        Explique comment une requête sera exécutée
        
        Args:
            query: Requête à expliquer
            
        Returns:
            Explication détaillée
        """
        try:
            # Optimisation de la requête
            optimized_query = await self.performance_optimizer.optimize_query(query)
            
            # Construction de la requête Elasticsearch
            es_query = await self.query_executor._build_elasticsearch_query(optimized_query)
            
            # Explication
            explanation = {
                "original_query": {
                    "text": query.query,
                    "type": query.search_type,
                    "filters": len(query.filters) if query.filters else 0,
                    "aggregations": len(query.aggregations) if query.aggregations else 0
                },
                "optimized_query": {
                    "text": optimized_query.query,
                    "type": optimized_query.search_type,
                    "size": optimized_query.size,
                    "timeout": optimized_query.timeout
                },
                "elasticsearch_query": {
                    "index": es_query.index,
                    "body_preview": str(es_query.body)[:500] + "..." if len(str(es_query.body)) > 500 else str(es_query.body)
                },
                "template_used": self.template_manager.get_template_name_for_query(query),
                "optimization_level": self.optimization_level.value,
                "cache_key": self.performance_optimizer._generate_cache_key(optimized_query)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'explication de la requête: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Vérifie la santé du moteur lexical
        
        Returns:
            État de santé détaillé
        """
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {}
            }
            
            # Vérification Elasticsearch
            try:
                es_health = await self.es_client.health_check()
                health_status["components"]["elasticsearch"] = {
                    "status": "healthy" if es_health["status"] == "ok" else "unhealthy",
                    "details": es_health
                }
            except Exception as e:
                health_status["components"]["elasticsearch"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Vérification des composants internes
            components = {
                "query_executor": self.query_executor,
                "result_processor": self.result_processor,
                "performance_optimizer": self.performance_optimizer,
                "template_manager": self.template_manager
            }
            
            for name, component in components.items():
                try:
                    # Test basique de fonctionnement
                    if hasattr(component, 'get_stats'):
                        component.get_stats()
                    health_status["components"][name] = {"status": "healthy"}
                except Exception as e:
                    health_status["components"][name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            
            # Test de requête simple
            try:
                test_query = SearchServiceQuery(
                    query="test",
                    size=1,
                    timeout="5s"
                )
                await self.search(test_query)
                health_status["components"]["search_functionality"] = {"status": "healthy"}
            except Exception as e:
                health_status["components"]["search_functionality"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def optimize_performance(self, level: OptimizationLevel):
        """
        Change le niveau d'optimisation du moteur
        
        Args:
            level: Nouveau niveau d'optimisation
        """
        try:
            old_level = self.optimization_level
            self.optimization_level = level
            
            # Recréation de l'optimiseur avec le nouveau niveau
            self.performance_optimizer = PerformanceOptimizer(optimization_level=level)
            
            # Réinitialisation du cache si niveau plus agressif
            if level.value == OptimizationLevel.AGGRESSIVE.value:
                await self._warm_cache()
            
            logger.info(f"✅ Niveau d'optimisation changé: {old_level.value} → {level.value}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du changement d'optimisation: {e}")
            raise
    
    async def clear_all_caches(self):
        """
        Vide tous les caches du moteur
        """
        try:
            # Cache de l'optimiseur
            self.performance_optimizer.clear_cache()
            
            # Cache du processeur de résultats
            self.result_processor.clear_cache()
            
            # Cache du template manager
            if hasattr(self.template_manager, 'clear_cache'):
                self.template_manager.clear_cache()
            
            logger.info("🧹 Tous les caches vidés")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du vidage des caches: {e}")
            raise
    
    async def rebuild_index_templates(self):
        """
        Reconstruit les templates d'index Elasticsearch
        """
        try:
            await self.template_manager.rebuild_templates()
            logger.info("🔄 Templates d'index reconstruits")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la reconstruction des templates: {e}")
            raise
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """
        Valide la configuration du moteur
        
        Returns:
            Rapport de validation
        """
        try:
            validation_report = {
                "status": "valid",
                "checks": {},
                "warnings": [],
                "errors": []
            }
            
            # Vérification de la connexion Elasticsearch
            try:
                await self.es_client.health_check()
                validation_report["checks"]["elasticsearch_connection"] = "✅ OK"
            except Exception as e:
                validation_report["checks"]["elasticsearch_connection"] = f"❌ {str(e)}"
                validation_report["errors"].append(f"Connexion Elasticsearch: {e}")
                validation_report["status"] = "invalid"
            
            # Vérification des templates
            try:
                template_count = await self.template_manager.get_template_count()
                if template_count > 0:
                    validation_report["checks"]["templates"] = f"✅ {template_count} templates chargés"
                else:
                    validation_report["checks"]["templates"] = "⚠️ Aucun template chargé"
                    validation_report["warnings"].append("Aucun template de requête disponible")
            except Exception as e:
                validation_report["checks"]["templates"] = f"❌ {str(e)}"
                validation_report["errors"].append(f"Templates: {e}")
            
            # Vérification de la configuration du cache
            cache_stats = self.performance_optimizer.get_cache_stats()
            if cache_stats["max_size"] > 0:
                validation_report["checks"]["cache"] = f"✅ Cache configuré ({cache_stats['max_size']} entrées max)"
            else:
                validation_report["checks"]["cache"] = "⚠️ Cache désactivé"
                validation_report["warnings"].append("Cache désactivé - performances réduites")
            
            # Vérification des métriques
            if self.enable_metrics and self.metrics_collector:
                validation_report["checks"]["metrics"] = "✅ Métriques activées"
            else:
                validation_report["checks"]["metrics"] = "⚠️ Métriques désactivées"
                validation_report["warnings"].append("Métriques désactivées - monitoring limité")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la validation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def export_configuration(self) -> Dict[str, Any]:
        """
        Exporte la configuration actuelle du moteur
        
        Returns:
            Configuration exportée
        """
        try:
            config = {
                "engine_version": "1.0.0",
                "export_timestamp": datetime.now().isoformat(),
                "optimization_level": self.optimization_level.value,
                "elasticsearch_config": self.es_client.get_connection_info(),
                "performance_settings": {
                    "cache_size": self.performance_optimizer.query_cache.maxsize,
                    "query_timeout": self.performance_optimizer.query_timeout,
                    "batch_size": self.performance_optimizer.batch_size
                },
                "enabled_features": {
                    "metrics": self.enable_metrics,
                    "cache_warming": self.performance_optimizer.warm_cache_enabled
                },
                "optimization_rules": [
                    {
                        "name": rule.name,
                        "enabled": rule.enabled,
                        "priority": rule.priority
                    }
                    for rule in self.performance_optimizer.optimization_rules
                ]
            }
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'export de configuration: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """
        Arrêt propre du moteur
        """
        try:
            logger.info("🔄 Arrêt du moteur lexical...")
            
            # Sauvegarde des métriques si nécessaire
            if self.metrics_collector:
                await self.metrics_collector.flush_metrics()
            
            # Fermeture des connexions
            await self.es_client.close()
            
            # Nettoyage des caches
            await self.clear_all_caches()
            
            self.is_initialized = False
            
            logger.info("✅ Moteur lexical arrêté proprement")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt: {e}")
            raise
    
    def __str__(self) -> str:
        """
        Représentation string du moteur
        
        Returns:
            Description du moteur
        """
        return (
            f"LexicalEngine("
            f"initialized={self.is_initialized}, "
            f"optimization={self.optimization_level.value}, "
            f"metrics={self.enable_metrics}, "
            f"elasticsearch={self.es_client.get_connection_info()['host']}"
            f")"
        )
    
    def __repr__(self) -> str:
        """
        Représentation pour debug
        
        Returns:
            Représentation détaillée
        """
        return self.__str__()


class LexicalEngineFactory:
    """
    Factory pour créer des instances de LexicalEngine
    """
    
    @staticmethod
    async def create_engine(
        elasticsearch_config: Dict[str, Any],
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        enable_metrics: bool = True
    ) -> LexicalEngine:
        """
        Crée et initialise un moteur lexical
        
        Args:
            elasticsearch_config: Configuration Elasticsearch
            optimization_level: Niveau d'optimisation
            enable_metrics: Activer les métriques
            
        Returns:
            Moteur lexical initialisé
        """
        try:
            # Création du client Elasticsearch
            es_client = ElasticsearchClient(elasticsearch_config)
            await es_client.connect()
            
            # Création du moteur
            engine = LexicalEngine(
                elasticsearch_client=es_client,
                optimization_level=optimization_level,
                enable_metrics=enable_metrics
            )
            
            # Initialisation
            await engine.initialize()
            
            logger.info("✅ Moteur lexical créé et initialisé via Factory")
            
            return engine
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du moteur: {e}")
            raise
    
    @staticmethod
    def create_development_engine() -> LexicalEngine:
        """
        Crée un moteur pour le développement
        
        Returns:
            Moteur configuré pour le développement
        """
        config = {
            "host": "localhost",
            "port": 9200,
            "timeout": 30
        }
        
        return LexicalEngineFactory.create_engine(
            elasticsearch_config=config,
            optimization_level=OptimizationLevel.BASIC,
            enable_metrics=True
        )
    
    @staticmethod
    def create_production_engine(elasticsearch_config: Dict[str, Any]) -> LexicalEngine:
        """
        Crée un moteur pour la production
        
        Args:
            elasticsearch_config: Configuration Elasticsearch
            
        Returns:
            Moteur configuré pour la production
        """
        return LexicalEngineFactory.create_engine(
            elasticsearch_config=elasticsearch_config,
            optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_metrics=True
        )
    
    async def _warm_cache(self):
        """
        Préchauffe le cache avec des requêtes communes
        """
        try:
            # Requêtes communes pour le préchauffage
            common_queries = [
                SearchServiceQuery(query="", search_type="match_all", size=20),
                SearchServiceQuery(query="financial", size=10),
                SearchServiceQuery(query="report", size=10),
                SearchServiceQuery(query="analysis", size=10),
            ]
            
            self.performance_optimizer.warm_cache(
                common_queries,
                self.query_executor.execute_query
            )
            
            logger.info("🔥 Cache préchauffé avec les requêtes communes")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du préchauffage du cache: {e}")
    
    async def _record_search_metrics(
        self, 
        query: SearchServiceQuery, 
        results: SearchServiceResponse, 
        execution_time
    ):
        """
        Enregistre les métriques de recherche
        
        Args:
            query: Requête exécutée
            results: Résultats obtenus
            execution_time: Temps d'exécution
        """
        try:
            if not self.metrics_collector:
                return
            
            metrics = {
                "operation": "search",
                "query_text": query.query,
                "query_type": query.search_type,
                "result_count": len(results.hits),
                "execution_time_ms": execution_time.total_seconds() * 1000,
                "has_aggregations": len(results.aggregations) > 0,
                "cache_hit": results.metadata.processing_time < 10,  # Heuristique
                "optimization_level": self.optimization_level.value
            }
            
            await self.metrics_collector.record_metrics("lexical_search", metrics)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement des métriques: {e}")
    
    async def _record_multi_search_metrics(
        self, 
        queries: List[SearchServiceQuery], 
        results: List[SearchServiceResponse], 
        execution_time
    ):
        """
        Enregistre les métriques de multi-recherche
        
        Args:
            queries: Requêtes exécutées
            results: Résultats obtenus
            execution_time: Temps d'exécution total
        """
        try:
            if not self.metrics_collector:
                return
            
            total_results = sum(len(result.hits) for result in results)
            
            metrics = {
                "operation": "multi_search",
                "query_count": len(queries),
                "total_results": total_results,
                "execution_time_ms": execution_time.total_seconds() * 1000,
                "avg_time_per_query": (execution_time.total_seconds() * 1000) / len(queries),
                "optimization_level": self.optimization_level.value
            }
            
            await self.metrics_collector.record_metrics("lexical_multi_search", metrics)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement des métriques multi-recherche: {e}")
    
    async def _record_aggregation_metrics(
        self, 
        query: SearchServiceQuery, 
        results: SearchServiceResponse, 
        execution_time
    ):
        """
        Enregistre les métriques d'agrégation
        
        Args:
            query: Requête avec agrégations
            results: Résultats avec agrégations
            execution_time: Temps d'exécution
        """
        try:
            if not self.metrics_collector:
                return
            
            metrics = {
                "operation": "aggregation_search",
                "aggregation_count": len(results.aggregations),
                "execution_time_ms": execution_time.total_seconds() * 1000,
                "result_count": len(results.hits),
                "optimization_level": self.optimization_level.value
            }
            
            await self.metrics_collector.record_metrics("lexical_aggregation", metrics)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement des métriques d'agrégation: {e}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques complètes du moteur
        
        Returns:
            Statistiques détaillées
        """
        try:
            stats = {
                "engine_info": {
                    "is_initialized": self.is_initialized,
                    "startup_time_seconds": self.startup_time.total_seconds() if self.startup_time else None,
                    "optimization_level": self.optimization_level.value,
                    "metrics_enabled": self.enable_metrics
                },
                "elasticsearch_client": self.es_client.get_client_stats(),
                "performance_optimizer": self.performance_optimizer.get_performance_stats(),
                "result_processor": self.result_processor.get_processing_stats(),
                "query_executor": self.query_executor.get_query_stats(),
                "template_manager": self.template_manager.get_template_stats(),
                "cache_stats": self.performance_optimizer.get_cache_stats()
            }
            
            if self.metrics_collector:
                stats["metrics"] = self.metrics_collector.get_metrics_summary()
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
            return {"error": str(e)}