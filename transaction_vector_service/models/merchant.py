# transaction_vector_service/models/merchant.py
"""
Merchant data models.

This module defines the data structures for merchant entities,
which represent businesses and service providers in transactions.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4

from .schema import VectorizedData, AuditInfo


class MerchantBase(BaseModel):
    """Base merchant data shared between creation and reading."""
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    category_id: Optional[int] = None
    website: Optional[str] = None
    country_code: Optional[str] = None


class MerchantCreate(MerchantBase):
    """Model for creating a new merchant."""
    raw_patterns: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Merchant(MerchantBase, AuditInfo):
    """Complete merchant model with internal data."""
    id: UUID = Field(default_factory=uuid4)
    patterns: List[str] = []
    logo_url: Optional[str] = None
    parent_category_id: Optional[int] = None
    aliases: List[str] = []
    transaction_count: int = 0
    confidence_score: float = 1.0
    
    class Config:
        orm_mode = True


class MerchantVector(Merchant, VectorizedData):
    """Merchant with vector embedding data for pattern matching."""
    name_embedding: Optional[List[float]] = None
    pattern_embeddings: Optional[Dict[str, List[float]]] = None


class MerchantRead(BaseModel):
    """Public merchant model returned from API endpoints."""
    id: UUID
    name: str
    display_name: Optional[str]
    category_id: Optional[int]
    category_name: Optional[str] = None
    logo_url: Optional[str] = None
    website: Optional[str] = None
    country_code: Optional[str] = None
    transaction_count: int

    class Config:
        orm_mode = True


class MerchantDetail(MerchantRead):
    """Detailed merchant model with additional information."""
    description: Optional[str]
    parent_category_id: Optional[int]
    parent_category_name: Optional[str] = None
    aliases: List[str] = []
    patterns: List[str] = []
    created_at: datetime
    updated_at: datetime


class MerchantSummary(BaseModel):
    """Summary information about a merchant."""
    id: UUID
    name: str
    display_name: Optional[str]
    logo_url: Optional[str]
    category_name: Optional[str]
    transaction_count: int
    last_transaction_date: Optional[datetime]
    total_amount: float = 0


class MerchantStats(BaseModel):
    """Statistical summary of transactions with a specific merchant."""
    merchant_id: UUID
    merchant_name: str
    transaction_count: int
    total_spent: float
    average_transaction: float
    first_transaction: datetime
    last_transaction: datetime
    frequency: str  # "weekly", "monthly", etc.
    by_month: Dict[str, float]  # Month -> Amount