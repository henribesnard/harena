# Import tous les modèles pour qu'ils soient disponibles via db_service.models
from db_service.models.user import User, BridgeConnection, UserPreference
from db_service.models.sync import SyncItem, SyncAccount, LoanDetail, RawTransaction, BridgeCategory, RawStock, AccountInformation, BridgeInsight, SyncTask, SyncStat

# Exporter tous les modèles
__all__ = [
    'User', 'BridgeConnection', 'UserPreference',
    'SyncItem', 'SyncAccount', 'LoanDetail', 'RawTransaction', 'BridgeCategory',
    'RawStock', 'AccountInformation', 'BridgeInsight', 'SyncTask', 'SyncStat'
]