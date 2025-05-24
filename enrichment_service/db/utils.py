"""
Module de gestion de base de données pour le service d'enrichissement.

Ce module fournit l'accès aux données SQL et la gestion des triggers
PostgreSQL pour les mises à jour vectorielles en temps réel.
"""

from enrichment_service.db.session import get_db, get_db_context, db_manager
from enrichment_service.db.triggers.install import install_triggers, remove_triggers, check_triggers_installed
from enrichment_service.db.triggers.handlers import TriggerEventHandler
from enrichment_service.db.utils import (
    get_user_enrichment_readiness,
    get_transactions_for_enrichment,
    get_user_spending_patterns,
    find_duplicate_transactions,
    get_enrichment_performance_metrics
)

__all__ = [
    # Session et gestionnaire de base de données
    'get_db',
    'get_db_context', 
    'db_manager',
    
    # Triggers PostgreSQL
    'install_triggers',
    'remove_triggers',
    'check_triggers_installed',
    'TriggerEventHandler',
    
    # Utilitaires d'enrichissement
    'get_user_enrichment_readiness',
    'get_transactions_for_enrichment',
    'get_user_spending_patterns',
    'find_duplicate_transactions',
    'get_enrichment_performance_metrics'
]