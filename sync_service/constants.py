"""
Constantes partagées pour le service de synchronisation.

Ce module contient les constantes configurables utilisées dans le service.
"""
import os

# Bridge API Configuration
BRIDGE_MAX_PAGES = int(os.getenv("BRIDGE_MAX_PAGES", "50"))
BRIDGE_TIMEOUT = float(os.getenv("BRIDGE_TIMEOUT", "30.0"))
BRIDGE_LONG_TIMEOUT = float(os.getenv("BRIDGE_LONG_TIMEOUT", "60.0"))

# Pagination Configuration
DEFAULT_PAGE_LIMIT = int(os.getenv("DEFAULT_PAGE_LIMIT", "50"))
MAX_TRANSACTION_LIMIT = int(os.getenv("MAX_TRANSACTION_LIMIT", "500"))
MAX_STOCK_LIMIT = int(os.getenv("MAX_STOCK_LIMIT", "500"))

# Retry Configuration
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_MIN_WAIT_SECONDS = float(os.getenv("RETRY_MIN_WAIT_SECONDS", "2"))
RETRY_MAX_WAIT_SECONDS = float(os.getenv("RETRY_MAX_WAIT_SECONDS", "10"))

# Sync Configuration
SYNC_TIMEOUT = float(os.getenv("SYNC_TIMEOUT", "300"))  # 5 minutes par défaut
ES_SYNC_TIMEOUT = float(os.getenv("ES_SYNC_TIMEOUT", "300"))  # 5 minutes pour Elasticsearch
BUDGET_CALC_TIMEOUT = float(os.getenv("BUDGET_CALC_TIMEOUT", "180"))  # 3 minutes pour budget profiling
