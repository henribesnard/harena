"""
Module d'enrichissement des données financières.

Ce module contient tous les enrichisseurs spécialisés pour transformer
les données financières brutes en informations structurées et contextualisées.
"""

from enrichment_service.enrichers.transaction_enricher import TransactionEnricher
from enrichment_service.enrichers.merchant_normalizer import MerchantNormalizer, MerchantInfo
from enrichment_service.enrichers.pattern_detector import PatternDetector, TransactionPattern
from enrichment_service.enrichers.summary_generator import SummaryGenerator, FinancialSummary
from enrichment_service.enrichers.account_profiler import AccountProfiler, AccountProfile
from enrichment_service.enrichers.insight_generator import InsightGenerator, FinancialInsight

__all__ = [
    # Enrichisseurs principaux
    'TransactionEnricher',
    'MerchantNormalizer', 
    'PatternDetector',
    'SummaryGenerator',
    'AccountProfiler',
    'InsightGenerator',
    
    # Classes de données
    'MerchantInfo',
    'TransactionPattern',
    'FinancialSummary',
    'AccountProfile',
    'FinancialInsight'
]