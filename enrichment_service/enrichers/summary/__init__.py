"""
Module de génération de résumés financiers.

Ce module contient tous les composants nécessaires pour générer
des résumés financiers périodiques avec analyses et comparaisons.
"""

from enrichment_service.enrichers.summary.data_models import (
    FinancialSummary, 
    CategorySummary, 
    MerchantSummary
)
from enrichment_service.enrichers.summary.analyzer import FinancialAnalyzer
from enrichment_service.enrichers.summary.narrator import SummaryNarrator
from enrichment_service.enrichers.summary.comparator import SummaryComparator

__all__ = [
    'FinancialSummary',
    'CategorySummary', 
    'MerchantSummary',
    'FinancialAnalyzer',
    'SummaryNarrator',
    'SummaryComparator'
]