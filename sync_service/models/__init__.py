# fichier sync_service/models/__init__.py

# Importer les bases avant tout pour établir les métadonnées
from sync_service.models.base import Base, TimestampMixin

# Importer les modèles dans le bon ordre
from user_service.models.user import User, BridgeConnection, UserPreference
from sync_service.models.sync import (
    WebhookEvent, SyncItem, SyncAccount, LoanDetail, 
    RawTransaction, BridgeCategory, RawStock, 
    AccountInformation, BridgeInsight, SyncTask, SyncStat
)

# Exporter uniquement les modèles nécessaires
__all__ = [
    'Base', 'TimestampMixin',
    'WebhookEvent', 'SyncItem', 'SyncAccount', 'LoanDetail', 
    'RawTransaction', 'BridgeCategory', 'RawStock', 
    'AccountInformation', 'BridgeInsight', 'SyncTask', 'SyncStat'
]