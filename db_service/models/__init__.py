"""
Import tous les modèles pour qu'ils soient disponibles via db_service.models.
Inclut maintenant les modèles de conversation pour le service IA.
"""

# Import modèles utilisateur
from db_service.models.user import User, BridgeConnection, UserPreference

# Import modèles de synchronisation
from db_service.models.sync import (
    SyncItem, SyncAccount, LoanDetail, RawTransaction, BridgeCategory,
    RawStock, AccountInformation, BridgeInsight, SyncTask, SyncStat
)

# NOUVEAU : Import modèles de conversation IA
from db_service.models.conversation import (
    Conversation,
    ConversationTurn,
    ConversationSummary,
    ConversationMessage,
)

# Exporter tous les modèles
__all__ = [
    # Modèles utilisateur
    'User', 'BridgeConnection', 'UserPreference',
    
    # Modèles de synchronisation
    'SyncItem', 'SyncAccount', 'LoanDetail', 'RawTransaction', 'BridgeCategory',
    'RawStock', 'AccountInformation', 'BridgeInsight', 'SyncTask', 'SyncStat',
    
    # NOUVEAU : Modèles conversation IA
    'Conversation', 'ConversationTurn', 'ConversationSummary', 'ConversationMessage'
]