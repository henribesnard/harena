"""
Constantes partagées pour le service utilisateur.

Ce module contient les constantes configurables utilisées dans le service.
"""
import os

# Bridge API Configuration
BRIDGE_MAX_PAGES = int(os.getenv("BRIDGE_MAX_PAGES", "50"))
BRIDGE_TIMEOUT = float(os.getenv("BRIDGE_TIMEOUT", "30.0"))
BRIDGE_LONG_TIMEOUT = float(os.getenv("BRIDGE_LONG_TIMEOUT", "60.0"))

# Cache Configuration
CATEGORIES_CACHE_TTL_HOURS = int(os.getenv("CATEGORIES_CACHE_TTL_HOURS", "24"))

# Pagination Configuration
DEFAULT_PAGE_LIMIT = int(os.getenv("DEFAULT_PAGE_LIMIT", "200"))
MAX_TRANSACTION_LIMIT = int(os.getenv("MAX_TRANSACTION_LIMIT", "500"))

# Retry Configuration
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_MIN_WAIT_SECONDS = float(os.getenv("RETRY_MIN_WAIT_SECONDS", "2"))
RETRY_MAX_WAIT_SECONDS = float(os.getenv("RETRY_MAX_WAIT_SECONDS", "10"))
