"""
Common utilities and types package.

This package contains common types, constants, and utilities used
throughout the application to ensure consistency and avoid circular imports.
"""

from .types import (
    SearchMode,
    SortOrder,
    TransactionType,
    RecurrenceFrequency,
    MerchantInfo,
    CategoryInfo,
    SearchWeights,
    DateRange,
    AmountRange,
    GeoLocation,
    VectorSearchResult,
    TransactionStat,
    RecurringPattern,
    SimilarityScore,
    SearchExplanation,
    
    # Constants
    DEFAULT_VECTOR_DIMENSION,
    SIMILARITY_THRESHOLD,
    BM25_THRESHOLD,
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    SEARCH_BATCH_SIZE,
    MIN_RECURRING_OCCURRENCES,
    MAX_DATE_VARIANCE_DAYS,
    AMOUNT_VARIANCE_PERCENT,
    TRANSACTION_COLLECTION,
    MERCHANT_COLLECTION
)

__all__ = [
    "SearchMode",
    "SortOrder",
    "TransactionType",
    "RecurrenceFrequency",
    "MerchantInfo",
    "CategoryInfo",
    "SearchWeights",
    "DateRange",
    "AmountRange",
    "GeoLocation",
    "VectorSearchResult",
    "TransactionStat",
    "RecurringPattern",
    "SimilarityScore",
    "SearchExplanation",
    
    "DEFAULT_VECTOR_DIMENSION",
    "SIMILARITY_THRESHOLD",
    "BM25_THRESHOLD",
    "DEFAULT_SEARCH_LIMIT",
    "MAX_SEARCH_LIMIT",
    "SEARCH_BATCH_SIZE",
    "MIN_RECURRING_OCCURRENCES",
    "MAX_DATE_VARIANCE_DAYS",
    "AMOUNT_VARIANCE_PERCENT",
    "TRANSACTION_COLLECTION",
    "MERCHANT_COLLECTION"
]