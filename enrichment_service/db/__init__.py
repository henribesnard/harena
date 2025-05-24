"""
Module de gestion de base de données pour le service d'enrichissement.

Ce module fournit l'accès aux données SQL et la gestion des triggers
PostgreSQL pour les mises à jour vectorielles en temps réel.
"""

from enrichment_service.db.session import get_db, get_db_context
from enrichment_service.db.triggers.install import install_triggers, remove_triggers
from enrichment_service.db.triggers.handlers import TriggerEventHandler

__all__ = [
    'get_db',
    'get_db_context', 
    'install_triggers',
    'remove_triggers',
    'TriggerEventHandler'
]