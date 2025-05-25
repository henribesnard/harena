"""
Module d'insights financiers pour le service d'enrichissement.

Ce module contient tous les composants nécessaires pour générer,
analyser et gérer les insights financiers automatiques.
"""

# Modèles de données
from enrichment_service.enrichers.insights.data_models import (
    FinancialInsight,
    InsightType,
    TimeScope,
    FinancialScope,
    Priority,
    InsightMetrics,
    InsightContext,
    InsightAction,
    InsightTemplate,
    InsightAnalytics,
    InsightRecommendation,
    INSIGHT_TEMPLATES
)

# Analyseurs spécialisés
from enrichment_service.enrichers.insights.analyzers import (
    BaseInsightAnalyzer,
    SpendingAnalyzer,
    SavingsAnalyzer,
    TrendAnalyzer,
    AnomalyAnalyzer,
    OpportunityAnalyzer,
    BudgetAnalyzer,
    AlertAnalyzer
)

# Gestionnaire d'insights
from enrichment_service.enrichers.insights.manager import InsightManager

__all__ = [
    # Modèles de données
    'FinancialInsight',
    'InsightType',
    'TimeScope',
    'FinancialScope',
    'Priority',
    'InsightMetrics',
    'InsightContext',
    'InsightAction',
    'InsightTemplate',
    'InsightAnalytics',
    'InsightRecommendation',
    'INSIGHT_TEMPLATES',
    
    # Analyseurs
    'BaseInsightAnalyzer',
    'SpendingAnalyzer',
    'SavingsAnalyzer', 
    'TrendAnalyzer',
    'AnomalyAnalyzer',
    'OpportunityAnalyzer',
    'BudgetAnalyzer',
    'AlertAnalyzer',
    
    # Gestionnaire
    'InsightManager'
]