"""
Constantes partagées pour le service d'enrichissement.

Ce module contient les constantes configurables utilisées dans le service.
"""
import os

# Elasticsearch Configuration
ES_TIMEOUT = float(os.getenv("ES_TIMEOUT", "30.0"))
ES_BULK_TIMEOUT = float(os.getenv("ES_BULK_TIMEOUT", "60.0"))
ES_DEFAULT_BATCH_SIZE = int(os.getenv("ES_DEFAULT_BATCH_SIZE", "500"))
ES_MAX_BATCH_SIZE = int(os.getenv("ES_MAX_BATCH_SIZE", "1000"))

# Retry Configuration
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_MIN_WAIT_SECONDS = float(os.getenv("RETRY_MIN_WAIT_SECONDS", "2"))
RETRY_MAX_WAIT_SECONDS = float(os.getenv("RETRY_MAX_WAIT_SECONDS", "10"))

# Cache Configuration
ACCOUNT_CACHE_SIZE = int(os.getenv("ACCOUNT_CACHE_SIZE", "128"))
CATEGORY_CACHE_TTL_HOURS = int(os.getenv("CATEGORY_CACHE_TTL_HOURS", "1"))

# Connection Pool Configuration
ES_CONNECTION_LIMIT = int(os.getenv("ES_CONNECTION_LIMIT", "10"))
ES_CONNECTION_LIMIT_PER_HOST = int(os.getenv("ES_CONNECTION_LIMIT_PER_HOST", "5"))

# LLM Configuration
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30.0"))
LLM_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE", "100"))
