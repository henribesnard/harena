"""
Module templates du Search Service
Expose les moteurs de templates pour requêtes et agrégations financières
"""

from typing import Dict, List, Optional, Any, Union
import logging

# === IMPORTS DES TEMPLATES ===

# Templates de requêtes par intention
from .query_templates import (
    # Classes principales
    QueryTemplateEngine,
    FinancialQueryTemplateBuilder,
    QueryTemplateValidator,
    TemplatePerformanceOptimizer,
    TemplateVersionManager,
    TemplateMetrics,
    
    # Enums et types
    IntentType,
    TemplateParameterType,
    
    # Modèles
    QueryTemplate,
    TemplateParameter,
    
    # Fonctions utilitaires
    build_query_from_intent,
    get_available_templates,
    get_template_by_intent,
    get_performance_estimate,
    validate_template_compatibility,
    process_financial_query,
    
    # Hooks de monitoring
    before_query_execution,
    after_query_execution,
    
    # Instances globales
    financial_template_builder,
    template_validator,
    performance_optimizer as query_performance_optimizer,
    version_manager as query_version_manager,
    template_metrics as query_template_metrics
)

# Templates d'agrégation financière
from .aggregation_templates import (
    # Classes principales
    FinancialAggregationEngine,
    AggregationQueryBuilder,
    AggregationResultProcessor,
    AggregationComposer,
    AggregationOrchestrator,
    AggregationCache,
    AggregationPerformanceMonitor,
    
    # Enums et types
    AggregationIntent,
    AggregationComplexity,
    AggregationDimension,
    AggregationMetric,
    
    # Modèles
    AggregationTemplate,
    
    # Fonctions utilitaires
    execute_financial_aggregation,
    get_monthly_spending_breakdown,
    get_category_analysis,
    get_top_merchants,
    get_spending_statistics,
    compare_periods,
    
    # Instances globales
    aggregation_engine,
    aggregation_orchestrator
)

# Configuration et logging
logger = logging.getLogger(__name__)


# === CLASSE UNIFIÉE POUR LA GESTION DES TEMPLATES ===

class TemplateManager:
    """
    Gestionnaire unifié pour tous les types de templates
    Fournit une interface unique pour requêtes et agrégations
    """
    
    def __init__(self):
        self.query_engine = QueryTemplateEngine()
        self.aggregation_engine = FinancialAggregationEngine()
        self.query_builder = financial_template_builder
        self.aggregation_orchestrator = aggregation_orchestrator
        
        logger.info("TemplateManager initialisé avec succès")
    
    # === MÉTHODES POUR TEMPLATES DE REQUÊTES ===
    
    def build_search_query(self, contract) -> Any:
        """Construit une requête de recherche à partir d'un contrat"""
        try:
            return build_query_from_intent(contract)
        except Exception as e:
            logger.error(f"Erreur lors de la construction de requête: {str(e)}")
            raise
    
    def get_query_templates(self) -> List[QueryTemplate]:
        """Récupère tous les templates de requête disponibles"""
        return get_available_templates()
    
    def get_query_template_by_intent(self, intent_type: str) -> Optional[QueryTemplate]:
        """Récupère un template de requête par intention"""
        return get_template_by_intent(intent_type)
    
    def validate_query_compatibility(self, contract) -> bool:
        """Valide la compatibilité d'un contrat avec les templates"""
        return validate_template_compatibility(contract)
    
    def get_query_performance_estimate(self, intent_type: str) -> int:
        """Estime le temps d'exécution pour un type d'intention"""
        return get_performance_estimate(intent_type)
    
    # === MÉTHODES POUR TEMPLATES D'AGRÉGATION ===
    
    def execute_aggregation(self, intent: AggregationIntent, user_id: int,
                           filters: Optional[Dict[str, Any]] = None,
                           use_cache: bool = True) -> Dict[str, Any]:
        """Exécute une agrégation financière"""
        try:
            return execute_financial_aggregation(intent, user_id, filters, use_cache)
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution d'agrégation: {str(e)}")
            raise
    
    def get_aggregation_templates(self) -> List[Dict[str, Any]]:
        """Récupère tous les templates d'agrégation disponibles"""
        return self.aggregation_orchestrator.get_available_aggregations()
    
    def get_aggregation_template_by_intent(self, intent: AggregationIntent) -> Optional[AggregationTemplate]:
        """Récupère un template d'agrégation par intention"""
        return self.aggregation_engine.get_template(intent)
    
    # === MÉTHODES DE MONITORING ET PERFORMANCE ===
    
    def get_query_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques des templates de requête"""
        return {
            "most_used_templates": query_template_metrics.get_most_used_templates(),
            "performance_summary": query_template_metrics.get_performance_summary(),
            "total_executions": sum(query_template_metrics.usage_stats.values()),
            "error_count": sum(query_template_metrics.error_stats.values())
        }
    
    def get_aggregation_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques des templates d'agrégation"""
        return self.aggregation_orchestrator.get_performance_report()
    
    def get_unified_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance unifié"""
        return {
            "query_templates": self.get_query_metrics(),
            "aggregation_templates": self.get_aggregation_metrics(),
            "cache_statistics": {
                "aggregation_cache": self.aggregation_orchestrator.cache.get_cache_stats()
            },
            "system_health": self._get_system_health()
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Évalue la santé globale du système de templates"""
        query_templates_count = len(self.query_engine.list_templates())
        agg_templates_count = len(self.aggregation_engine.list_templates())
        
        # Calcul de métriques de santé
        total_query_executions = sum(query_template_metrics.usage_stats.values())
        total_query_errors = sum(query_template_metrics.error_stats.values())
        query_error_rate = (total_query_errors / total_query_executions) if total_query_executions > 0 else 0
        
        agg_cache_stats = self.aggregation_orchestrator.cache.get_cache_stats()
        
        return {
            "total_templates": query_templates_count + agg_templates_count,
            "query_templates_count": query_templates_count,
            "aggregation_templates_count": agg_templates_count,
            "query_error_rate": query_error_rate,
            "aggregation_cache_hit_rate": agg_cache_stats["hit_rate"],
            "system_status": "healthy" if query_error_rate < 0.05 and agg_cache_stats["hit_rate"] > 0.6 else "degraded"
        }
    
    # === MÉTHODES DE CACHE ===
    
    def clear_aggregation_cache(self, user_id: Optional[int] = None):
        """Vide le cache d'agrégation (optionnellement pour un utilisateur spécifique)"""
        if user_id:
            self.aggregation_orchestrator.cache.invalidate_user_cache(user_id)
            logger.info(f"Cache d'agrégation vidé pour l'utilisateur {user_id}")
        else:
            self.aggregation_orchestrator.cache.cache.clear()
            logger.info("Cache d'agrégation entièrement vidé")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Retourne le statut détaillé du cache"""
        agg_cache_stats = self.aggregation_orchestrator.cache.get_cache_stats()
        
        return {
            "aggregation_cache": {
                "enabled": True,
                "stats": agg_cache_stats,
                "recommendation": self._get_cache_recommendation(agg_cache_stats)
            }
        }
    
    def _get_cache_recommendation(self, cache_stats: Dict[str, Any]) -> str:
        """Génère une recommandation basée sur les stats de cache"""
        hit_rate = cache_stats["hit_rate"]
        
        if hit_rate > 0.8:
            return "Cache très efficace, aucune action nécessaire"
        elif hit_rate > 0.6:
            return "Cache efficace, surveillance recommandée"
        elif hit_rate > 0.4:
            return "Cache modérément efficace, optimisation possible"
        else:
            return "Cache peu efficace, révision des stratégies de cache nécessaire"


# === FONCTIONS D'UTILITÉ GLOBALES ===

def get_template_by_type_and_intent(template_type: str, intent: str) -> Optional[Union[QueryTemplate, AggregationTemplate]]:
    """
    Récupère un template par type (query/aggregation) et intention
    """
    if template_type == "query":
        return get_template_by_intent(intent)
    elif template_type == "aggregation":
        try:
            agg_intent = AggregationIntent(intent)
            return aggregation_engine.get_template(agg_intent)
        except ValueError:
            logger.warning(f"Intention d'agrégation inconnue: {intent}")
            return None
    else:
        logger.error(f"Type de template inconnu: {template_type}")
        return None


def get_all_available_templates() -> Dict[str, List[Dict[str, Any]]]:
    """
    Retourne tous les templates disponibles (requêtes et agrégations)
    """
    query_templates = [
        {
            "type": "query",
            "intent": t.intent_type.value,
            "name": t.template_name,
            "description": t.description,
            "performance_category": t.performance_category,
            "estimated_time_ms": t.estimated_time_ms
        }
        for t in get_available_templates()
    ]
    
    aggregation_templates = [
        {
            "type": "aggregation", 
            "intent": t.intent.value,
            "name": t.name,
            "description": t.description,
            "complexity": t.complexity.value,
            "estimated_time_ms": t.estimated_time_ms
        }
        for t in aggregation_engine.list_templates()
    ]
    
    return {
        "query_templates": query_templates,
        "aggregation_templates": aggregation_templates,
        "total_count": len(query_templates) + len(aggregation_templates)
    }


def execute_template_by_intent(intent_type: str, template_type: str, 
                              user_id: int, **kwargs) -> Dict[str, Any]:
    """
    Exécute un template par intention et type de façon unifiée
    """
    if template_type == "query":
        # Pour les requêtes, on a besoin d'un contrat complet
        contract = kwargs.get("contract")
        if not contract:
            raise ValueError("Contract requis pour l'exécution d'un template de requête")
        return {"elasticsearch_query": build_query_from_intent(contract)}
    
    elif template_type == "aggregation":
        try:
            agg_intent = AggregationIntent(intent_type)
            filters = kwargs.get("filters")
            use_cache = kwargs.get("use_cache", True)
            return execute_financial_aggregation(agg_intent, user_id, filters, use_cache)
        except ValueError:
            raise ValueError(f"Intention d'agrégation invalide: {intent_type}")
    
    else:
        raise ValueError(f"Type de template invalide: {template_type}")


def get_template_performance_report() -> Dict[str, Any]:
    """
    Génère un rapport de performance global pour tous les templates
    """
    manager = TemplateManager()
    return manager.get_unified_performance_report()


# === HOOKS D'INTÉGRATION ===

def initialize_templates_system():
    """
    Initialise le système de templates avec validation
    """
    try:
        manager = TemplateManager()
        
        # Validation des templates de requête
        query_templates = manager.get_query_templates()
        logger.info(f"Chargé {len(query_templates)} templates de requête")
        
        # Validation des templates d'agrégation
        agg_templates = manager.get_aggregation_templates()
        logger.info(f"Chargé {len(agg_templates)} templates d'agrégation")
        
        # Test de connectivité
        health = manager._get_system_health()
        logger.info(f"Système de templates initialisé - Status: {health['system_status']}")
        
        return manager
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des templates: {str(e)}")
        raise


def shutdown_templates_system():
    """
    Arrêt propre du système de templates
    """
    try:
        # Vider les caches
        aggregation_orchestrator.cache.cache.clear()
        
        # Log des statistiques finales
        final_metrics = get_template_performance_report()
        logger.info(f"Système de templates arrêté - Métriques finales: {final_metrics}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt des templates: {str(e)}")


# === INSTANCE GLOBALE ===
template_manager = TemplateManager()


# === EXPORTS PRINCIPAUX ===

__all__ = [
    # === CLASSES UNIFIÉES ===
    "TemplateManager",
    
    # === CLASSES DE REQUÊTE ===
    "QueryTemplateEngine",
    "FinancialQueryTemplateBuilder", 
    "QueryTemplateValidator",
    "TemplatePerformanceOptimizer",
    "TemplateVersionManager",
    "TemplateMetrics",
    
    # === CLASSES D'AGRÉGATION ===
    "FinancialAggregationEngine",
    "AggregationQueryBuilder",
    "AggregationResultProcessor", 
    "AggregationComposer",
    "AggregationOrchestrator",
    "AggregationCache",
    "AggregationPerformanceMonitor",
    
    # === ENUMS ET TYPES ===
    # Requêtes
    "IntentType",
    "TemplateParameterType",
    # Agrégations
    "AggregationIntent",
    "AggregationComplexity",
    "AggregationDimension",
    "AggregationMetric",
    
    # === MODÈLES ===
    "QueryTemplate",
    "TemplateParameter", 
    "AggregationTemplate",
    
    # === FONCTIONS DE REQUÊTE ===
    "build_query_from_intent",
    "get_available_templates",
    "get_template_by_intent",
    "get_performance_estimate",
    "validate_template_compatibility",
    "process_financial_query",
    
    # === FONCTIONS D'AGRÉGATION ===
    "execute_financial_aggregation",
    "get_monthly_spending_breakdown",
    "get_category_analysis", 
    "get_top_merchants",
    "get_spending_statistics",
    "compare_periods",
    
    # === FONCTIONS UNIFIÉES ===
    "get_template_by_type_and_intent",
    "get_all_available_templates",
    "execute_template_by_intent",
    "get_template_performance_report",
    
    # === HOOKS SYSTÈME ===
    "initialize_templates_system",
    "shutdown_templates_system",
    "before_query_execution",
    "after_query_execution",
    
    # === INSTANCES GLOBALES ===
    "template_manager",
    "financial_template_builder", 
    "template_validator",
    "query_performance_optimizer",
    "query_version_manager",
    "query_template_metrics",
    "aggregation_engine",
    "aggregation_orchestrator"
]


# === HELPERS D'IMPORT ===

def get_query_components():
    """Retourne les composants de templates de requête"""
    return {
        "engine": QueryTemplateEngine(),
        "builder": financial_template_builder,
        "validator": template_validator,
        "optimizer": query_performance_optimizer,
        "metrics": query_template_metrics
    }


def get_aggregation_components():
    """Retourne les composants de templates d'agrégation"""
    return {
        "engine": aggregation_engine,
        "orchestrator": aggregation_orchestrator,
        "cache": aggregation_orchestrator.cache,
        "monitor": aggregation_orchestrator.monitor,
        "composer": AggregationComposer(aggregation_engine)
    }


def get_all_components():
    """Retourne tous les composants du système de templates"""
    return {
        "manager": template_manager,
        "query": get_query_components(),
        "aggregation": get_aggregation_components()
    }


# === INITIALISATION AUTOMATIQUE ===

try:
    # Initialisation automatique du système au chargement du module
    logger.info("Initialisation automatique du système de templates...")
    template_manager = initialize_templates_system()
    logger.info("Système de templates prêt à l'utilisation")
    
except Exception as e:
    logger.error(f"Échec de l'initialisation automatique des templates: {str(e)}")
    # Fallback vers une instance basique
    template_manager = TemplateManager()
    logger.warning("Système de templates en mode dégradé")


# === MÉTADONNÉES DU MODULE ===

__version__ = "1.0.0"
__author__ = "Search Service Team"
__description__ = "Système unifié de templates pour requêtes et agrégations financières"
__license__ = "MIT"