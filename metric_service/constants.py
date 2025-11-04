"""
Constantes partagées pour le service de métriques.

Ce module contient les constantes configurables utilisées dans le service.
"""
import os

# Cache Configuration
CACHE_TTL_SECONDS = int(os.getenv("METRIC_CACHE_TTL", "300"))  # 5 minutes par défaut
CACHE_TIMEOUT = float(os.getenv("CACHE_TIMEOUT", "5.0"))

# Database Query Configuration
DB_QUERY_TIMEOUT = float(os.getenv("DB_QUERY_TIMEOUT", "30.0"))
MAX_MONTHS_ANALYSIS = int(os.getenv("MAX_MONTHS_ANALYSIS", "24"))

# Redis Configuration
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))

# Retry Configuration
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_MIN_WAIT_SECONDS = float(os.getenv("RETRY_MIN_WAIT_SECONDS", "1"))
RETRY_MAX_WAIT_SECONDS = float(os.getenv("RETRY_MAX_WAIT_SECONDS", "5"))

# Forecast Configuration
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "30"))
MIN_TRANSACTIONS_FOR_FORECAST = int(os.getenv("MIN_TRANSACTIONS_FOR_FORECAST", "10"))
