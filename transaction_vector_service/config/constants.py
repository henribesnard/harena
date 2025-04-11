# transaction_vector_service/config/constants.py
"""
Constants used throughout the Transaction Vector Service.

This module defines constants that are used across the application
to maintain consistency and avoid duplication.
"""

# Qdrant collection names
TRANSACTION_COLLECTION = "transactions"
MERCHANT_COLLECTION = "merchants"

# Transaction search constants
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 200
SIMILARITY_THRESHOLD = 0.75

# Category hierarchy constants
CATEGORY_LEVELS = {
    "L1": "primary",
    "L2": "secondary",
    "L3": "tertiary"
}

# Transaction type mappings
TRANSACTION_TYPES = {
    "card": "Carte bancaire",
    "transfer": "Virement",
    "direct_debit": "Prélèvement",
    "check": "Chèque",
    "cash": "Espèces",
    "loan_payment": "Remboursement de prêt",
    "fee": "Frais bancaires",
    "income": "Revenu",
    "refund": "Remboursement",
    "unknown": "Opération inconnue"
}

# Recurring transaction detection constants
MIN_RECURRING_OCCURRENCES = 3
MAX_DATE_VARIANCE_DAYS = 5
AMOUNT_VARIANCE_PERCENT = 0.1

# Insight generation constants
INSIGHT_TIMEFRAMES = ["weekly", "monthly", "quarterly", "yearly"]
MIN_TRANSACTIONS_FOR_INSIGHTS = 5

# Cache TTL values (in seconds)
CATEGORY_CACHE_TTL = 86400  # 24 hours
MERCHANT_CACHE_TTL = 3600  # 1 hour
EMBEDDING_CACHE_TTL = 86400 * 7  # 7 days

# API rate limiting
API_RATE_LIMIT = 100  # requests per minute
API_RATE_LIMIT_PERIOD = 60  # seconds

# Language support
SUPPORTED_LANGUAGES = ["fr", "en"]
DEFAULT_LANGUAGE = "fr"

# Currency formatting
DEFAULT_CURRENCY = "EUR"
CURRENCY_SYMBOLS = {
    "EUR": "€",
    "USD": "$",
    "GBP": "£"
}

# Date format strings
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"