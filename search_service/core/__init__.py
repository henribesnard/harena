"""
Module core du Search Service - Moteur de recherche lexicale Elasticsearch
================================================================================

Ce module fournit les composants principaux pour la recherche lexicale haute performance :
- LexicalSearchEngine : Moteur de recherche principal
- QueryExecutionEngine : Exécuteur de requêtes optimisé  
- ResultProcessor : Traitement avancé des résultats
- PerformanceOptimizer : Optimisations de performance

Architecture :
    User Query → LexicalEngine → QueryExecutor → Elasticsearch → ResultProcessor → Response

Utilisé par :
    - API routes pour les recherches lexicales
    - Conversation Service via les contrats standardisés
    - Tests et monitoring des performances
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
from datetime import datetime

# === IMPORTS DES COMPOSANTS CORE ===

from .lexical_engine import (
    LexicalSearchEngine,
    LexicalEngineManager,
    SearchMode,
    FieldBoostStrategy,
    QueryOptimization,
    LexicalSearchConfig,
    FinancialFieldConfiguration,
    SearchContext,
    # Fonctions utilitaires
    analyze_query_complexity,
    optimize_query_for_performance,
    # Instance globale
    lexical_engine_manager
)

from .query_executor import (
    QueryExecutionEngine,
    HighLevelQueryExecutor,
    QueryExecutorManager,
    ExecutionStrategy,
    QueryComplexity,
    ExecutionPriority,
    ExecutionContext,
    ExecutionResult,
    # Fonctions principales
    execute_search_query,
    execute_aggregation_query,
    batch_execute_queries,
    initialize_query_executor,
    get_execution_health,
    shutdown_query_executor,
    # Instance globale
    query_executor_manager
)

from .result_processor import (
    ResultProcessor,
    ProcessingStrategy,
    ProcessingContext,
    ResultEnrichment,
    ProcessorManager,
    # Fonctions de traitement
    process_search_results,
    enrich_financial_results,
    format_for_response,
    # Instance globale
    result_processor_manager
)

from .performance_optimizer import (
    PerformanceOptimizer,
    PerformanceMetrics,
    OptimizationLevel,
    PerformanceProfile,
    QueryComplexity as PerfQueryComplexity,
    OptimizationType,
    PerformanceConfiguration,
    CacheOptimizer,
    QueryRewriter,
    TimeoutManager,
    BatchProcessor
)


# === CONFIGURATION DU LOGGING ===

logger = logging.getLogger(__name__)

# Loggers spécialisés pour chaque composant
lexical_logger = logging.getLogger(f"{__name__}.lexical_engine")
execution_logger = logging.getLogger(f"{__name__}.query_executor")
processing_logger = logging.getLogger(f"{__name__}.result_processor")
performance_logger = logging.getLogger(f"{__name__}.performance_optimizer")


# === GESTIONNAIRE CORE GLOBAL ===

class CoreComponentManager:
    """Gestionnaire centralisé des composants core du Search Service"""
    
    def __init__(self):
        self._initialized = False
        self._lexical_engine: Optional[LexicalEngineManager] = None
        self._query_executor: Optional[QueryExecutorManager] = None
        self._result_processor: Optional[ProcessorManager] = None
        self._performance_optimizer: Optional[PerformanceOptimizer] = None
        self._startup_time: Optional[datetime] = None
        
        logger.info("CoreComponentManager créé")
    
    async def initialize(
        self,
        elasticsearch_client,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialise tous les composants core du Search Service
        
        Args:
            elasticsearch_client: Client Elasticsearch configuré
            config: Configuration optionnelle pour les composants
            
        Returns:
            Dict contenant le statut d'initialisation de chaque composant
        """
        if self._initialized:
            logger.warning("CoreComponentManager déjà initialisé")
            return await self.get_initialization_status()
        
        initialization_results = {}
        
        try:
            logger.info("Initialisation des composants core du Search Service...")
            
            # 1. Initialiser le gestionnaire de performance d'abord
            perf_config = PerformanceConfiguration()
            if config and "performance" in config:
                perf_config.__dict__.update(config["performance"])
            
            self._performance_optimizer = PerformanceOptimizer(perf_config)
            initialization_results["performance_optimizer"] = "initialized"
            logger.info("✅ PerformanceOptimizer initialisé")
            
            # 2. Initialiser l'exécuteur de requêtes
            executor_config = config.get("query_executor", {}) if config else {}
            self._query_executor = query_executor_manager
            await self._query_executor.initialize(elasticsearch_client, executor_config)
            initialization_results["query_executor"] = "initialized"
            logger.info("✅ QueryExecutorManager initialisé")
            
            # 3. Initialiser le processeur de résultats
            processor_config = config.get("result_processor", {}) if config else {}
            self._result_processor = result_processor_manager
            await self._result_processor.initialize(processor_config)
            initialization_results["result_processor"] = "initialized"
            logger.info("✅ ResultProcessorManager initialisé")
            
            # 4. Initialiser le moteur lexical principal
            lexical_config = config.get("lexical_engine", {}) if config else {}
            self._lexical_engine = lexical_engine_manager
            await self._lexical_engine.initialize(
                elasticsearch_client=elasticsearch_client,
                performance_optimizer=self._performance_optimizer,
                **lexical_config
            )
            initialization_results["lexical_engine"] = "initialized"
            logger.info("✅ LexicalEngineManager initialisé")
            
            # 5. Vérifier la connectivité et la santé
            health_checks = await self._run_health_checks()
            initialization_results["health_checks"] = health_checks
            
            # 6. Marquer comme initialisé
            self._initialized = True
            self._startup_time = datetime.now()
            
            logger.info("🚀 Tous les composants core sont initialisés avec succès")
            
            return {
                "status": "success",
                "initialized_at": self._startup_time.isoformat(),
                "components": initialization_results
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation des composants core: {e}")
            
            # Nettoyer les composants partiellement initialisés
            await self._cleanup_partial_initialization()
            
            return {
                "status": "failed",
                "error": str(e),
                "components": initialization_results
            }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Arrêt propre de tous les composants"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        shutdown_results = {}
        
        try:
            logger.info("Arrêt des composants core...")
            
            # Arrêter dans l'ordre inverse de l'initialisation
            if self._lexical_engine:
                await self._lexical_engine.shutdown()
                shutdown_results["lexical_engine"] = "shutdown"
            
            if self._result_processor:
                await self._result_processor.shutdown()
                shutdown_results["result_processor"] = "shutdown"
            
            if self._query_executor:
                await self._query_executor.shutdown()
                shutdown_results["query_executor"] = "shutdown"
            
            # Le performance optimizer n'a pas besoin d'arrêt explicite
            shutdown_results["performance_optimizer"] = "shutdown"
            
            self._initialized = False
            logger.info("✅ Tous les composants core arrêtés")
            
            return {
                "status": "success",
                "shutdown_at": datetime.now().isoformat(),
                "components": shutdown_results
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt des composants: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "components": shutdown_results
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Vérification de santé globale du système core"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "error": "Core components not initialized"
            }
        
        try:
            health_status = {
                "system_status": "healthy",
                "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
                "components": {}
            }
            
            # Vérifier chaque composant
            if self._lexical_engine:
                health_status["components"]["lexical_engine"] = await self._lexical_engine.health_check()
            
            if self._query_executor:
                health_status["components"]["query_executor"] = await self._query_executor.health_check()
            
            if self._result_processor:
                health_status["components"]["result_processor"] = await self._result_processor.health_check()
            
            if self._performance_optimizer:
                health_status["components"]["performance_optimizer"] = {
                    "status": "healthy",
                    "summary": self._performance_optimizer.get_performance_summary()
                }
            
            # Déterminer le statut global
            component_statuses = [
                comp_health.get("status", "unknown") 
                for comp_health in health_status["components"].values()
            ]
            
            if all(status == "healthy" for status in component_statuses):
                health_status["system_status"] = "healthy"
            elif any(status == "error" for status in component_statuses):
                health_status["system_status"] = "degraded"
            else:
                health_status["system_status"] = "partial"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de santé: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance global"""
        if not self._initialized:
            return {"error": "System not initialized"}
        
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "system_uptime_hours": (datetime.now() - self._startup_time).total_seconds() / 3600,
                "components": {}
            }
            
            # Métriques du moteur lexical
            if self._lexical_engine:
                report["components"]["lexical_engine"] = self._lexical_engine.get_performance_report()
            
            # Métriques de l'exécuteur
            if self._query_executor:
                report["components"]["query_executor"] = await self._query_executor.get_performance_metrics()
            
            # Métriques du processeur
            if self._result_processor:
                report["components"]["result_processor"] = await self._result_processor.get_performance_metrics()
            
            # Métriques de l'optimiseur
            if self._performance_optimizer:
                report["components"]["performance_optimizer"] = self._performance_optimizer.get_performance_summary()
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")
            return {"error": str(e)}
    
    async def _run_health_checks(self) -> Dict[str, Any]:
        """Lance les vérifications de santé initiales"""
        health_results = {}
        
        try:
            # Test de connectivité Elasticsearch via l'exécuteur
            if self._query_executor:
                es_health = await self._query_executor.test_elasticsearch_connection()
                health_results["elasticsearch"] = es_health
            
            # Test du moteur lexical
            if self._lexical_engine:
                lexical_health = await self._lexical_engine.health_check()
                health_results["lexical_engine"] = lexical_health
            
            # Test du processeur
            if self._result_processor:
                processor_health = await self._result_processor.health_check()
                health_results["result_processor"] = processor_health
            
            return health_results
            
        except Exception as e:
            logger.error(f"Erreur lors des vérifications de santé: {e}")
            return {"error": str(e)}
    
    async def _cleanup_partial_initialization(self):
        """Nettoie les composants partiellement initialisés"""
        try:
            if self._lexical_engine and self._lexical_engine._initialized:
                await self._lexical_engine.shutdown()
            
            if self._query_executor and self._query_executor._initialized:
                await self._query_executor.shutdown()
            
            if self._result_processor and self._result_processor._initialized:
                await self._result_processor.shutdown()
            
            logger.info("Nettoyage des composants partiellement initialisés terminé")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
    
    async def get_initialization_status(self) -> Dict[str, Any]:
        """Retourne le statut d'initialisation détaillé"""
        return {
            "initialized": self._initialized,
            "startup_time": self._startup_time.isoformat() if self._startup_time else None,
            "components": {
                "lexical_engine": self._lexical_engine._initialized if self._lexical_engine else False,
                "query_executor": self._query_executor._initialized if self._query_executor else False,
                "result_processor": self._result_processor._initialized if self._result_processor else False,
                "performance_optimizer": self._performance_optimizer is not None
            }
        }
    
    @property
    def lexical_engine(self) -> Optional[LexicalEngineManager]:
        """Accès au gestionnaire du moteur lexical"""
        return self._lexical_engine if self._initialized else None
    
    @property
    def query_executor(self) -> Optional[QueryExecutorManager]:
        """Accès au gestionnaire d'exécution de requêtes"""
        return self._query_executor if self._initialized else None
    
    @property
    def result_processor(self) -> Optional[ProcessorManager]:
        """Accès au gestionnaire de traitement des résultats"""
        return self._result_processor if self._initialized else None
    
    @property
    def performance_optimizer(self) -> Optional[PerformanceOptimizer]:
        """Accès à l'optimiseur de performance"""
        return self._performance_optimizer if self._initialized else None


# === INSTANCE GLOBALE ===

core_manager = CoreComponentManager()


# === FONCTIONS D'INTERFACE PUBLIQUE ===

async def initialize_core_components(
    elasticsearch_client,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Initialise tous les composants core du Search Service
    
    Point d'entrée principal pour l'initialisation du système
    """
    return await core_manager.initialize(elasticsearch_client, config)


async def shutdown_core_components() -> Dict[str, Any]:
    """Arrêt propre de tous les composants core"""
    return await core_manager.shutdown()


async def get_core_health() -> Dict[str, Any]:
    """Vérification de santé globale des composants core"""
    return await core_manager.get_system_health()


async def get_core_performance() -> Dict[str, Any]:
    """Rapport de performance global des composants core"""
    return await core_manager.get_performance_report()


def get_lexical_engine() -> Optional[LexicalEngineManager]:
    """Accès sécurisé au moteur lexical"""
    return core_manager.lexical_engine


def get_query_executor() -> Optional[QueryExecutorManager]:
    """Accès sécurisé à l'exécuteur de requêtes"""
    return core_manager.query_executor


def get_result_processor() -> Optional[ProcessorManager]:
    """Accès sécurisé au processeur de résultats"""
    return core_manager.result_processor


def get_performance_optimizer() -> Optional[PerformanceOptimizer]:
    """Accès sécurisé à l'optimiseur de performance"""
    return core_manager.performance_optimizer


# === FONCTIONS UTILITAIRES INTÉGRÉES ===

async def execute_lexical_search(search_request) -> Dict[str, Any]:
    """
    Interface simplifiée pour les recherches lexicales
    
    Combine automatiquement l'optimisation, l'exécution et le traitement
    """
    if not core_manager._initialized:
        raise RuntimeError("Core components not initialized. Call initialize_core_components() first.")
    
    try:
        # Utiliser le moteur lexical principal
        lexical_engine = core_manager.lexical_engine
        return await lexical_engine.search(search_request)
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche lexicale: {e}")
        raise


async def optimize_and_execute_query(
    query_body: Dict[str, Any],
    search_request,
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED
) -> Tuple[Dict[str, Any], List[OptimizationType]]:
    """
    Optimise et exécute une requête avec rapport des optimisations appliquées
    """
    if not core_manager._initialized:
        raise RuntimeError("Core components not initialized")
    
    # Optimiser la requête
    optimizer = core_manager.performance_optimizer
    optimized_query, optimizations = await optimizer.optimize_query(
        query_body, search_request, performance_profile
    )
    
    # Exécuter la requête optimisée
    executor = core_manager.query_executor
    result = await executor.execute_search(optimized_query, search_request)
    
    return result, optimizations


def get_system_status() -> Dict[str, str]:
    """Statut simple du système pour les health checks"""
    if not core_manager._initialized:
        return {"status": "not_ready", "reason": "components_not_initialized"}
    
    try:
        # Vérification rapide
        components_ready = all([
            core_manager._lexical_engine is not None,
            core_manager._query_executor is not None,
            core_manager._result_processor is not None,
            core_manager._performance_optimizer is not None
        ])
        
        if components_ready:
            return {"status": "ready"}
        else:
            return {"status": "partial", "reason": "some_components_missing"}
            
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === GESTIONNAIRE PRINCIPAL ===
    "CoreComponentManager",
    "core_manager",
    
    # === FONCTIONS D'INITIALISATION ===
    "initialize_core_components",
    "shutdown_core_components",
    "get_core_health",
    "get_core_performance",
    
    # === ACCÈS AUX COMPOSANTS ===
    "get_lexical_engine",
    "get_query_executor", 
    "get_result_processor",
    "get_performance_optimizer",
    
    # === FONCTIONS INTÉGRÉES ===
    "execute_lexical_search",
    "optimize_and_execute_query",
    "get_system_status",
    
    # === CLASSES ET ENUMS DES COMPOSANTS ===
    # Lexical Engine
    "LexicalSearchEngine",
    "LexicalEngineManager", 
    "SearchMode",
    "FieldBoostStrategy",
    "QueryOptimization",
    "LexicalSearchConfig",
    "FinancialFieldConfiguration",
    "SearchContext",
    
    # Query Executor
    "QueryExecutionEngine",
    "HighLevelQueryExecutor",
    "QueryExecutorManager",
    "ExecutionStrategy",
    "QueryComplexity",
    "ExecutionPriority",
    "ExecutionContext",
    "ExecutionResult",
    
    # Result Processor
    "ResultProcessor",
    "ProcessingStrategy",
    "ProcessingContext", 
    "ResultEnrichment",
    "ProcessorManager",
    
    # Performance Optimizer
    "PerformanceOptimizer",
    "PerformanceMetrics",
    "OptimizationLevel",
    "PerformanceProfile",
    "OptimizationType",
    "PerformanceConfiguration",
    "CacheOptimizer",
    "QueryRewriter",
    "TimeoutManager",
    "BatchProcessor",
    
    # === INSTANCES GLOBALES ===
    "lexical_engine_manager",
    "query_executor_manager",
    "result_processor_manager",
    
    # === FONCTIONS UTILITAIRES ===
    "analyze_query_complexity",
    "optimize_query_for_performance",
    "execute_search_query",
    "execute_aggregation_query",
    "batch_execute_queries",
    "process_search_results",
    "enrich_financial_results",
    "format_for_response"
]


# === INFORMATIONS DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Composants core du Search Service pour la recherche lexicale Elasticsearch"

# Logging de l'import du module
logger.info(f"Module core initialisé - version {__version__}")