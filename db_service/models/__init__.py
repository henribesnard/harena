# db_service/models/__init__.py
"""
Import tous les modèles pour qu'ils soient disponibles via db_service.models.
Version SIMPLIFIÉE avec seulement les modèles existants.
"""

# Import modèles utilisateur
from db_service.models.user import User, BridgeConnection, UserPreference

# Import modèles de synchronisation
from db_service.models.sync import (
    SyncItem, SyncAccount, LoanDetail, RawTransaction, Category,
    RawStock, AccountInformation, BridgeInsight, SyncTask, SyncStat
)

# Import modèles conversation SIMPLIFIÉS (seulement ceux qui existent)
from db_service.models.conversation import (
    Conversation,
    ConversationTurn
)

# Exporter seulement les modèles qui existent réellement
__all__ = [
    # Modèles utilisateur
    'User', 'BridgeConnection', 'UserPreference',
    
    # Modèles de synchronisation
    'SyncItem', 'SyncAccount', 'LoanDetail', 'RawTransaction', 'Category',
    'RawStock', 'AccountInformation', 'BridgeInsight', 'SyncTask', 'SyncStat',
    
    # Modèles conversation SIMPLES (seulement ceux qui existent)
    'Conversation', 'ConversationTurn'
]