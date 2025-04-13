"""
Transaction data models.

This module defines the data structures for banking transactions,
including raw transaction data, vectorized transactions, and search models.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Set, Tuple
from datetime import datetime, date
from uuid import UUID, uuid4
from enum import Enum

from .schema import VectorizedData, AuditInfo, Metadata


class SearchMode(str, Enum):
    """Search mode for hybrid search."""
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


class SearchWeights(BaseModel):
    """Weights for hybrid search components."""
    bm25_weight: float = Field(0.3, ge=0, le=1)
    vector_weight: float = Field(0.3, ge=0, le=1)
    cross_encoder_weight: float = Field(0.4, ge=0, le=1)

    @model_validator(mode='after')
    def check_weights_sum(self) -> 'SearchWeights':
        """Validate that weights sum to 1."""
        weights_sum = (
            self.bm25_weight + 
            self.vector_weight + 
            self.cross_encoder_weight
        )
        if abs(weights_sum - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError("Search weights must sum to 1.0")
        return self


class TransactionBase(BaseModel):
    """Base transaction data shared between creation and reading."""
    bridge_transaction_id: int
    account_id: int
    user_id: int
    amount: float
    currency_code: str = "EUR"
    description: str
    clean_description: Optional[str] = None
    transaction_date: date
    value_date: Optional[date] = None
    booking_date: Optional[date] = None
    category_id: Optional[int] = None
    operation_type: Optional[str] = None
    is_future: bool = False
    is_deleted: bool = False


class TransactionCreate(TransactionBase):
    """Model for creating a new transaction."""
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    

class Transaction(TransactionBase, AuditInfo):
    """Complete transaction model with internal data."""
    id: UUID = Field(default_factory=uuid4)
    normalized_merchant: Optional[str] = None
    merchant_id: Optional[UUID] = None
    parent_category_id: Optional[int] = None
    geo_info: Optional[Dict[str, Any]] = None
    metadata: Metadata = Field(default_factory=Metadata)
    
    model_config = {"from_attributes": True}


class TransactionVector(Transaction, VectorizedData):
    """Transaction with vector embedding data for semantic search."""
    description_embedding: Optional[List[float]] = None
    fingerprint: Optional[str] = None
    similar_transactions: Optional[List[UUID]] = None
    similarity_score: Optional[float] = None
    is_recurring: bool = False
    recurring_pattern_id: Optional[UUID] = None


class TransactionRead(BaseModel):
    """Public transaction model returned from API endpoints."""
    id: UUID
    amount: float
    currency_code: str
    description: str
    clean_description: Optional[str]
    transaction_date: date
    category_id: Optional[int]
    category_name: Optional[str] = None
    parent_category_id: Optional[int] = None
    parent_category_name: Optional[str] = None
    operation_type: Optional[str]
    normalized_merchant: Optional[str]
    merchant_logo_url: Optional[str] = None
    is_recurring: bool = False
    recurring_group_id: Optional[UUID] = None
    created_at: datetime
    # Relevance scores for search results
    relevance_score: Optional[float] = None
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    matched_terms: Optional[List[str]] = None

    model_config = {"from_attributes": True}


class TransactionDetail(TransactionRead):
    """Detailed transaction model with additional information."""
    account_id: int
    account_name: Optional[str] = None
    value_date: Optional[date]
    booking_date: Optional[date]
    geo_info: Optional[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similar_transactions: Optional[List[UUID]] = None
    updated_at: datetime


class TransactionSearch(BaseModel):
    """Model for transaction search requests."""
    query: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    categories: Optional[List[int]] = None
    merchant_names: Optional[List[str]] = None
    operation_types: Optional[List[str]] = None
    account_ids: Optional[List[int]] = None
    include_future: bool = False
    include_deleted: bool = False
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)
    sort_by: str = "transaction_date"
    sort_order: str = "desc"
    # Search options for hybrid search
    search_mode: SearchMode = SearchMode.HYBRID
    search_weights: Optional[SearchWeights] = None
    min_relevance: Optional[float] = Field(None, ge=0, le=1)
    include_explanation: bool = False

    @field_validator('end_date')
    @classmethod
    def end_date_must_be_after_start_date(cls, v: Optional[date], info: Dict[str, Any]) -> Optional[date]:
        """Validate date range."""
        if v and 'start_date' in info.data and info.data['start_date'] and v < info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

    @field_validator('max_amount')
    @classmethod
    def max_amount_must_be_greater_than_min_amount(cls, v: Optional[float], info: Dict[str, Any]) -> Optional[float]:
        """Validate amount range."""
        if v and 'min_amount' in info.data and info.data['min_amount'] is not None and v < info.data['min_amount']:
            raise ValueError('max_amount must be greater than min_amount')
        return v


class TransactionBatchCreate(BaseModel):
    """Model for creating multiple transactions in a batch."""
    transactions: List[TransactionCreate]
    sync_id: Optional[str] = None


class TransactionStats(BaseModel):
    """Statistical summary of transactions."""
    total_count: int
    total_amount: float
    average_amount: float
    min_amount: float
    max_amount: float
    date_range: Dict[str, date]
    by_category: Dict[str, Dict[str, Any]]
    by_month: Dict[str, Dict[str, Any]]


class TransactionRelevanceExplanation(BaseModel):
    """Explanation of search result relevance for a transaction."""
    matched_terms: List[str] = Field(default_factory=list)
    term_matches: Dict[str, int] = Field(default_factory=dict)
    semantic_similarity: float = 0.0
    contextual_relevance: float = 0.0
    relevance_factors: Dict[str, float] = Field(default_factory=dict)
    position_boost: float = 0.0


class TransactionSearchResults(BaseModel):
    """Model for search results with metadata."""
    results: List[TransactionRead]
    total: int
    page: int
    page_size: int
    has_more: bool
    query: Optional[str] = None
    search_mode: Optional[str] = None
    execution_time_ms: Optional[int] = None
    explanation: Optional[Dict[str, Any]] = None
    filters_applied: Dict[str, Any] = Field(default_factory=dict)


class TransactionSearchExplanation(BaseModel):
    """Explanation of search results for debugging and transparency."""
    query: str
    query_analysis: Dict[str, Any] = Field(default_factory=dict)
    search_mode: str
    weights: Dict[str, float] = Field(default_factory=dict)
    top_matching_terms: List[Tuple[str, int]] = Field(default_factory=list)
    category_distribution: List[Tuple[str, int]] = Field(default_factory=list)
    merchant_distribution: List[Tuple[str, int]] = Field(default_factory=list)
    date_range: Dict[str, Any] = Field(default_factory=dict)
    relevance_factors: Dict[str, str] = Field(default_factory=dict)
    
    model_config = {"arbitrary_types_allowed": True}  # Pour permettre l'utilisation de Tuple