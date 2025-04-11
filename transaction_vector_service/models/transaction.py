# transaction_vector_service/models/transaction.py
"""
Transaction data models.

This module defines the data structures for banking transactions,
including raw transaction data, vectorized transactions, and search models.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID, uuid4

from .schema import VectorizedData, AuditInfo, Metadata


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
    
    class Config:
        orm_mode = True


class TransactionVector(Transaction, VectorizedData):
    """Transaction with vector embedding data for semantic search."""
    description_embedding: Optional[List[float]] = None
    fingerprint: Optional[str] = None
    similar_transactions: Optional[List[UUID]] = None
    similarity_score: Optional[float] = None


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

    class Config:
        orm_mode = True


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

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        """Validate date range."""
        if v and 'start_date' in values and values['start_date'] and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

    @validator('max_amount')
    def max_amount_must_be_greater_than_min_amount(cls, v, values):
        """Validate amount range."""
        if v and 'min_amount' in values and values['min_amount'] is not None and v < values['min_amount']:
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