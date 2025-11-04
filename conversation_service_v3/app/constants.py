"""
Constantes partagées pour le service de conversation.

Ce module contient les constantes configurables utilisées dans le service.
"""
import os

# HTTP Configuration
HTTP_CLIENT_TIMEOUT = float(os.getenv("HTTP_CLIENT_TIMEOUT", "60.0"))
SEARCH_SERVICE_TIMEOUT = float(os.getenv("SEARCH_SERVICE_TIMEOUT", "60.0"))
BUDGET_SERVICE_TIMEOUT = float(os.getenv("BUDGET_SERVICE_TIMEOUT", "5.0"))

# Retry Configuration
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_MIN_WAIT_SECONDS = float(os.getenv("RETRY_MIN_WAIT_SECONDS", "1"))
RETRY_MAX_WAIT_SECONDS = float(os.getenv("RETRY_MAX_WAIT_SECONDS", "5"))

# LLM Configuration
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_TEMPERATURE_DEFAULT = float(os.getenv("LLM_TEMPERATURE_DEFAULT", "0.1"))
MAX_CORRECTION_ATTEMPTS = int(os.getenv("MAX_CORRECTION_ATTEMPTS", "2"))

# Agent Configuration
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))
MAX_TRANSACTIONS_IN_CONTEXT = int(os.getenv("MAX_TRANSACTIONS_IN_CONTEXT", "50"))

# Conversation Memory
MAX_CONVERSATION_MESSAGES = int(os.getenv("MAX_CONVERSATION_MESSAGES", "10"))
MAX_CONVERSATION_CONTEXT_TOKENS = int(os.getenv("MAX_CONVERSATION_CONTEXT_TOKENS", "4000"))
CONVERSATION_CACHE_TTL_SECONDS = int(os.getenv("CONVERSATION_CACHE_TTL_SECONDS", "86400"))  # 24h

# Redis Configuration
REDIS_CONNECTION_TIMEOUT = float(os.getenv("REDIS_CONNECTION_TIMEOUT", "5.0"))
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))

# Analytics
MIN_TRANSACTIONS_FOR_ANALYSIS = int(os.getenv("MIN_TRANSACTIONS_FOR_ANALYSIS", "5"))
SIGNIFICANCE_THRESHOLD_PERCENT = float(os.getenv("SIGNIFICANCE_THRESHOLD_PERCENT", "5.0"))
