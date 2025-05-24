"""
Gestionnaires de triggers PostgreSQL pour le service d'enrichissement.

Ce module contient les fonctionnalités pour installer et gérer les triggers
PostgreSQL qui permettent la mise à jour vectorielle en temps réel.
"""

from enrichment_service.db.triggers.install import install_triggers, remove_triggers, check_triggers_installed
from enrichment_service.db.triggers.handlers import TriggerEventHandler

__all__ = [
    'install_triggers',
    'remove_triggers', 
    'check_triggers_installed',
    'TriggerEventHandler'
]