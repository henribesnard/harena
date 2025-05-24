"""
Modèles de base de données pour le service d'enrichissement.

Ce module importe et réexporte tous les modèles nécessaires depuis db_service
pour faciliter l'accès aux données dans le service d'enrichissement.
"""

# Import de tous les modèles depuis db_service
from db_service.models.user import User, BridgeConnection, UserPreference
from db_service.models.sync import (
    SyncItem, SyncAccount, LoanDetail, RawTransaction, BridgeCategory, 
    RawStock, AccountInformation, BridgeInsight, SyncTask, SyncStat,
    WebhookEvent
)

# Modèles pour l'enrichissement (si spécifiques au service)
# from enrichment_service.db.models.enrichment import EnrichedTransaction, FinancialPattern

__all__ = [
    # Modèles utilisateur
    'User',
    'BridgeConnection', 
    'UserPreference',
    
    # Modèles de synchronisation
    'SyncItem',
    'SyncAccount',
    'LoanDetail',
    'RawTransaction',
    'BridgeCategory',
    'RawStock',
    'AccountInformation',
    'BridgeInsight',
    'SyncTask',
    'SyncStat',
    'WebhookEvent',
    
    # Modèles d'enrichissement (à ajouter si nécessaire)
    # 'EnrichedTransaction',
    # 'FinancialPattern'
]