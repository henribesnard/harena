"""
📝 Module Templates - Templates de requêtes et agrégations

Point d'entrée simplifié pour tous les templates du Search Service.
Expose les moteurs de templates pour requêtes et agrégations financières.
"""

# === IMPORTS TEMPLATES DE REQUÊTES ===
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
   performance_optimizer,
   version_manager,
   template_metrics
)

# === IMPORTS TEMPLATES D'AGRÉGATION ===
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

# === CLASSE GESTIONNAIRE SIMPLIFIÉE ===
class TemplateManager:
   """
   Gestionnaire unifié pour tous les types de templates
   
   Centralise l'accès aux templates de requêtes et d'agrégations.
   """
   def __init__(self):
       self.query_engine = QueryTemplateEngine()
       self.aggregation_engine = FinancialAggregationEngine()
       self.query_builder = financial_template_builder
       self.aggregation_orchestrator = aggregation_orchestrator
       self.performance_optimizer = performance_optimizer
       self.template_validator = template_validator
   
   async def initialize(self):
       """Initialise les templates"""
       # Initialisation des templates de requêtes
       self.query_templates = self.query_engine.list_templates()
       
       # Initialisation des templates d'agrégation
       self.aggregation_templates = self.aggregation_engine.list_templates()
   
   def get_query_template(self, intent_type: str):
       """Récupère un template de requête"""
       return get_template_by_intent(intent_type)
   
   def get_aggregation_template(self, intent: AggregationIntent):
       """Récupère un template d'agrégation"""
       return self.aggregation_engine.get_template(intent)
   
   def build_query_from_intent(self, contract):
       """Construit une requête depuis un contrat"""
       return build_query_from_intent(contract)
   
   def execute_aggregation(self, intent, user_id, filters=None):
       """Exécute une agrégation"""
       return execute_financial_aggregation(intent, user_id, filters)

# === INSTANCE GLOBALE ===
template_manager = TemplateManager()

# === FONCTIONS UNIFIÉES ===
def get_template_by_type_and_intent(template_type: str, intent: str):
   """Récupère un template par type et intention"""
   if template_type == "query":
       return get_template_by_intent(intent)
   elif template_type == "aggregation":
       try:
           agg_intent = AggregationIntent(intent)
           return aggregation_engine.get_template(agg_intent)
       except ValueError:
           return None
   return None

def get_all_available_templates():
   """Récupère tous les templates disponibles"""
   return {
       "query_templates": get_available_templates(),
       "aggregation_templates": aggregation_engine.list_templates()
   }

def execute_template_by_intent(template_type: str, intent: str, **kwargs):
   """Exécute un template par type et intention"""
   if template_type == "query":
       # Nécessite un contrat pour les requêtes
       contract = kwargs.get("contract")
       if contract:
           return build_query_from_intent(contract)
   elif template_type == "aggregation":
       # Exécution d'agrégation
       user_id = kwargs.get("user_id")
       filters = kwargs.get("filters")
       if user_id:
           try:
               agg_intent = AggregationIntent(intent)
               return execute_financial_aggregation(agg_intent, user_id, filters)
           except ValueError:
               return {"error": f"Unknown aggregation intent: {intent}"}
   
   return {"error": "Invalid template execution parameters"}

def get_template_performance_report():
   """Génère un rapport de performance des templates"""
   return {
       "query_performance": template_metrics.get_performance_summary(),
       "aggregation_performance": aggregation_orchestrator.get_performance_report(),
       "most_used_queries": template_metrics.get_most_used_templates(),
       "generated_at": "now"
   }

# === FONCTIONS D'INITIALISATION ===
async def initialize_templates_system():
   """Initialise le système de templates"""
   await template_manager.initialize()
   return template_manager

async def shutdown_templates_system():
   """Arrête le système de templates"""
   # Nettoyage si nécessaire
   pass

# === EXPORTS ===
__all__ = [
   # === GESTIONNAIRE PRINCIPAL ===
   "TemplateManager",
   "template_manager",
   
   # === TEMPLATES DE REQUÊTES ===
   # Classes principales
   "QueryTemplateEngine",
   "FinancialQueryTemplateBuilder",
   "QueryTemplateValidator",
   "TemplatePerformanceOptimizer",
   "TemplateVersionManager",
   "TemplateMetrics",
   
   # Enums requêtes
   "IntentType",
   "TemplateParameterType",
   
   # Modèles requêtes
   "QueryTemplate",
   "TemplateParameter",
   
   # Fonctions requêtes
   "build_query_from_intent",
   "get_available_templates",
   "get_template_by_intent",
   "get_performance_estimate",
   "validate_template_compatibility",
   "process_financial_query",
   
   # === TEMPLATES D'AGRÉGATION ===
   # Classes principales
   "FinancialAggregationEngine",
   "AggregationQueryBuilder",
   "AggregationResultProcessor",
   "AggregationComposer",
   "AggregationOrchestrator",
   "AggregationCache",
   "AggregationPerformanceMonitor",
   
   # Enums agrégations
   "AggregationIntent",
   "AggregationComplexity",
   "AggregationDimension",
   "AggregationMetric",
   
   # Modèles agrégations
   "AggregationTemplate",
   
   # Fonctions agrégations
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
   "financial_template_builder",
   "template_validator",
   "performance_optimizer",
   "version_manager",
   "template_metrics",
   "aggregation_engine",
   "aggregation_orchestrator"
]