"""
🔧 Module Core - Moteur de recherche et logique métier
"""
from datetime import datetime
# === IMPORTS MOTEUR ===
from .lexical_engine import (
    LexicalSearchEngine,
    LexicalEngineManager,
    lexical_engine_manager
)

# === IMPORTS EXECUTEUR ===
try:
    from .query_executor import QueryExecutor
except ImportError:
    # Créer une classe placeholder si non disponible
    class QueryExecutor:
        def __init__(self):
            pass

# === IMPORTS PROCESSEUR ===
from .result_processor import (
    FinancialResultProcessor,
    AggregationResultProcessor,
    ResultProcessorManager,
    result_processor_manager
)

# === IMPORTS OPTIMISEUR ===
try:
    from .performance_optimizer import PerformanceOptimizer
except ImportError:
    # Créer une classe placeholder si non disponible
    class PerformanceOptimizer:
        def __init__(self):
            pass


# === FONCTION UTILITAIRE PRINCIPALE ===
def get_lexical_engine() -> LexicalSearchEngine:
    """
    Retourne l'instance du moteur de recherche lexicale
    
    Returns:
        LexicalSearchEngine: Instance du moteur initialisé
        
    Raises:
        RuntimeError: Si le moteur n'est pas initialisé
    """
    return lexical_engine_manager.engine


# === FONCTIONS UTILITAIRES SUPPLÉMENTAIRES ===
def get_query_executor():
    """Retourne l'exécuteur de requêtes"""
    try:
        from .query_executor import query_executor_manager
        return query_executor_manager.executor if query_executor_manager._initialized else None
    except ImportError:
        return None


def get_result_processor():
    """Retourne le processeur de résultats"""
    return result_processor_manager


def get_performance_optimizer():
    """Retourne l'optimiseur de performance"""
    try:
        from .performance_optimizer import PerformanceOptimizer, PerformanceConfiguration
        config = PerformanceConfiguration()
        return PerformanceOptimizer(config)
    except ImportError:
        return None


async def get_core_health():
    """Retourne l'état de santé du core"""
    health_data = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    try:
        # Santé du moteur lexical
        lexical_health = await lexical_engine_manager.health_check()
        health_data["components"]["lexical_engine"] = lexical_health
        
        # Santé de l'exécuteur de requêtes
        query_executor = get_query_executor()
        if query_executor:
            try:
                from .query_executor import query_executor_manager
                executor_health = await query_executor_manager.health_check()
                health_data["components"]["query_executor"] = executor_health
            except:
                health_data["components"]["query_executor"] = {"status": "unknown"}
        else:
            health_data["components"]["query_executor"] = {"status": "not_available"}
        
        # Santé du processeur de résultats
        processor_stats = result_processor_manager.get_processing_statistics()
        health_data["components"]["result_processor"] = {
            "status": processor_stats.get("system_status", "unknown"),
            "statistics": processor_stats
        }
        
        # Déterminer le statut global
        component_statuses = []
        for component, data in health_data["components"].items():
            if isinstance(data, dict) and "status" in data:
                component_statuses.append(data["status"])
        
        if all(status in ["healthy", "active"] for status in component_statuses):
            health_data["status"] = "healthy"
        elif any(status in ["healthy", "active"] for status in component_statuses):
            health_data["status"] = "degraded"
        else:
            health_data["status"] = "unhealthy"
            
    except Exception as e:
        health_data["status"] = "error"
        health_data["error"] = str(e)
    
    return health_data


def get_core_performance():
    """Retourne les métriques de performance du core"""
    performance_data = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    try:
        # Performance du moteur lexical
        if lexical_engine_manager._initialized:
            lexical_perf = lexical_engine_manager.get_performance_report()
            performance_data["components"]["lexical_engine"] = lexical_perf
        else:
            performance_data["components"]["lexical_engine"] = {"status": "not_initialized"}
        
        # Performance de l'exécuteur de requêtes
        query_executor = get_query_executor()
        if query_executor:
            try:
                executor_stats = query_executor.get_execution_statistics()
                performance_data["components"]["query_executor"] = executor_stats
            except:
                performance_data["components"]["query_executor"] = {"status": "unknown"}
        else:
            performance_data["components"]["query_executor"] = {"status": "not_available"}
        
        # Performance du processeur de résultats
        processor_stats = result_processor_manager.get_processing_statistics()
        performance_data["components"]["result_processor"] = processor_stats
        
        # Performance de l'optimiseur
        optimizer = get_performance_optimizer()
        if optimizer:
            try:
                optimizer_perf = optimizer.get_performance_summary()
                performance_data["components"]["performance_optimizer"] = optimizer_perf
            except:
                performance_data["components"]["performance_optimizer"] = {"status": "unknown"}
        else:
            performance_data["components"]["performance_optimizer"] = {"status": "not_available"}
        
        # Résumé global
        total_components = len(performance_data["components"])
        active_components = sum(1 for comp in performance_data["components"].values() 
                              if isinstance(comp, dict) and comp.get("status") not in ["not_available", "unknown"])
        
        performance_data["summary"] = {
            "total_components": total_components,
            "active_components": active_components,
            "health_ratio": active_components / total_components if total_components > 0 else 0
        }
        
    except Exception as e:
        performance_data["error"] = str(e)
        performance_data["status"] = "error"
    
    return performance_data


# === CLASSE GESTIONNAIRE PRINCIPAL ===
class CoreManager:
    """Gestionnaire simplifié pour le core"""
    
    def __init__(self):
        self.lexical_engine_manager = lexical_engine_manager
        self.result_processor_manager = result_processor_manager
        self._query_executor = None
        self._performance_optimizer = None
        self._initialized = False
    
    @property
    def lexical_engine(self):
        """Accès au moteur de recherche lexicale"""
        return self.lexical_engine_manager.engine
    
    @property 
    def query_executor(self):
        """Accès à l'exécuteur de requêtes"""
        if self._query_executor is None:
            self._query_executor = get_query_executor()
        return self._query_executor
    
    @property
    def result_processor(self):
        """Accès au processeur de résultats"""
        return self.result_processor_manager
    
    @property
    def performance_optimizer(self):
        """Accès à l'optimiseur de performance"""
        if self._performance_optimizer is None:
            self._performance_optimizer = get_performance_optimizer()
        return self._performance_optimizer
    
    def get_search_engine(self):
        """
        Retourne le moteur de recherche lexicale
        
        Cette méthode est appelée par routes.py et fournit l'interface
        pour accéder au moteur de recherche depuis l'API.
        
        Returns:
            LexicalSearchEngine: Instance du moteur de recherche si initialisé
            None: Si le manager ou le moteur n'est pas initialisé
        """
        if not self.is_initialized():
            return None
        
        return self.lexical_engine
    
    async def initialize(self, elasticsearch_client=None):
        """Initialise tous les composants du core"""
        if self._initialized:
            return
        
        try:
            # Initialiser le moteur lexical si un client ES est fourni
            if elasticsearch_client:
                self.lexical_engine_manager.initialize(elasticsearch_client)
                
                # Initialiser aussi l'exécuteur de requêtes si disponible
                try:
                    from .query_executor import initialize_query_executor
                    initialize_query_executor(elasticsearch_client)
                except ImportError:
                    pass
            
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Erreur initialisation CoreManager: {str(e)}")
    
    def is_initialized(self) -> bool:
        """Vérifie si le manager est initialisé"""
        return self._initialized and self.lexical_engine_manager._initialized
    
    async def health_check(self):
        """Vérification de santé globale du core"""
        return await get_core_health()


# === INSTANCE GLOBALE ===
core_manager = CoreManager()


# === EXPORTS PRINCIPAUX ===
__all__ = [
    # Gestionnaire principal
    "CoreManager",
    "core_manager",
    
    # Composants principaux  
    "LexicalSearchEngine",
    "LexicalEngineManager",
    "FinancialResultProcessor",
    "AggregationResultProcessor",
    "ResultProcessorManager",
    "QueryExecutor", 
    "PerformanceOptimizer",
    
    # Instances globales
    "lexical_engine_manager",
    "result_processor_manager",
    
    # Fonctions utilitaires
    "get_lexical_engine",
    "get_query_executor",
    "get_result_processor", 
    "get_performance_optimizer",
    "get_core_health",
    "get_core_performance"
]


# === COMPATIBILITÉ RÉTROGRADE ===
# Alias pour maintenir la compatibilité
LexicalEngine = LexicalSearchEngine
ResultProcessor = FinancialResultProcessor


# === HELPERS POUR DÉBOGAGE ===
def get_core_status():
    """Retourne le statut détaillé du module core"""
    return {
        "core_manager_initialized": core_manager.is_initialized(),
        "lexical_engine_available": lexical_engine_manager._initialized,
        "result_processor_available": True,
        "components_loaded": {
            "lexical_engine": "LexicalSearchEngine" in globals(),
            "result_processor": "FinancialResultProcessor" in globals(),
            "query_executor": "QueryExecutor" in globals(),
            "performance_optimizer": "PerformanceOptimizer" in globals()
        }
    }


def debug_imports():
    """Fonction de débogage pour les imports"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        from .lexical_engine import LexicalSearchEngine
        logger.info("✅ LexicalSearchEngine importé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur import LexicalSearchEngine: {e}")
    
    try:
        from .result_processor import FinancialResultProcessor
        logger.info("✅ FinancialResultProcessor importé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur import FinancialResultProcessor: {e}")
    
    return get_core_status()


# === INITIALISATION ===
import logging
logger = logging.getLogger(__name__)
logger.info("📦 Module core initialisé avec succès")