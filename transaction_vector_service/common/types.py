"""
Common types and data structures.

This module defines common types, data structures, and constants used
throughout the Transaction Vector Service to ensure consistency and
avoid circular imports.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, NamedTuple
from enum import Enum
from datetime import date, datetime
from uuid import UUID
from pydantic import BaseModel


class SearchMode(str, Enum):
    """Search mode for transaction searches."""
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class TransactionType(str, Enum):
    """Types of financial transactions."""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    REFUND = "refund"
    UNKNOWN = "unknown"


class RecurrenceFrequency(str, Enum):
    """Frequencies for recurring transactions."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"
    IRREGULAR = "irregular"


class MerchantInfo(TypedDict, total=False):
    """Information about a merchant."""
    id: str
    name: str
    display_name: str
    category_id: Optional[int]
    score: float
    logo_url: Optional[str]


class CategoryInfo(TypedDict, total=False):
    """Information about a transaction category."""
    id: int
    name: str
    display_name: Optional[str]
    parent_id: Optional[int]
    level: str
    keywords: List[str]


class SearchWeights(BaseModel):
    """Weights for hybrid search components."""
    bm25_weight: float = 0.3
    vector_weight: float = 0.3
    cross_encoder_weight: float = 0.4
    
    class Config:
        validate_assignment = True


class DateRange(NamedTuple):
    """Date range for filtering transactions."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class AmountRange(NamedTuple):
    """Amount range for filtering transactions."""
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None


class GeoLocation(TypedDict, total=False):
    """Geographical location information."""
    country_code: str
    city: Optional[str]
    postal_code: Optional[str]
    coordinates: Optional[List[float]]


class VectorSearchResult(TypedDict):
    """Result from a vector similarity search."""
    id: str
    score: float
    payload: Dict[str, Any]


class TransactionStat(TypedDict, total=False):
    """Statistical information about transactions."""
    total_count: int
    total_amount: float
    average_amount: float
    min_amount: float
    max_amount: float
    date_range: Dict[str, Any]
    by_category: Dict[str, Dict[str, Any]]
    by_month: Dict[str, Dict[str, Any]]


class RecurringPattern(TypedDict, total=False):
    """Pattern for recurring transactions."""
    id: str
    merchant_name: str
    frequency: str
    typical_amount: float
    amount_variance: float
    transaction_count: int
    first_occurrence: str
    last_occurrence: str
    confidence_score: float
    expected_next_date: str


class SimilarityScore(TypedDict):
    """Similarity scores for different search methods."""
    bm25_score: float
    vector_score: float
    cross_encoder_score: float
    combined_score: float


class SearchExplanation(TypedDict, total=False):
    """Explanation of search results."""
    query: str
    result_count: int
    search_method: str
    matching_terms: List[str]
    top_categories: List[str]
    date_range: Dict[str, Any]
    merchant_distribution: Dict[str, int]
    relevance_factors: Dict[str, str]


# Constants for search and processing
DEFAULT_VECTOR_DIMENSION = 1536
SIMILARITY_THRESHOLD = 0.75
BM25_THRESHOLD = 0.2
DEFAULT_SEARCH_LIMIT = 50
MAX_SEARCH_LIMIT = 200
SEARCH_BATCH_SIZE = 100
MIN_RECURRING_OCCURRENCES = 3
MAX_DATE_VARIANCE_DAYS = 5
AMOUNT_VARIANCE_PERCENT = 0.1

# Collections for vector search
TRANSACTION_COLLECTION = "transactions"
MERCHANT_COLLECTION = "merchants"

# Cache TTL values (in seconds)
CATEGORY_CACHE_TTL = 86400  # 24 hours