# transaction_vector_service/models/schema.py
"""
Common schemas used across multiple models.

This module defines reusable schemas and validation utilities 
that are shared between multiple model definitions.
"""

from pydantic import BaseModel, Field, validator
from typing import Generic, TypeVar, List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum

# Generic type for paginated responses
T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response model."""
    items: List[T]
    total: int
    page: int
    page_size: int
    has_more: bool


class TimeRange(BaseModel):
    """Time range for filtering data."""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        """Validate that end_date is after start_date if both are provided."""
        if v and 'start_date' in values and values['start_date'] and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class AmountRange(BaseModel):
    """Amount range for filtering by transaction amounts."""
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    
    @validator('max_amount')
    def max_amount_greater_than_min(cls, v, values):
        """Validate that max_amount is greater than min_amount if both are provided."""
        if v is not None and 'min_amount' in values and values['min_amount'] is not None:
            if v < values['min_amount']:
                raise ValueError('max_amount must be greater than min_amount')
        return v


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


class SearchQuery(BaseModel):
    """Common search query parameters."""
    query: Optional[str] = None
    time_range: Optional[TimeRange] = None
    amount_range: Optional[AmountRange] = None
    categories: Optional[List[int]] = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESC
    include_details: bool = False


class VectorizedData(BaseModel):
    """Base model for data that has been vectorized."""
    vector_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    vector_updated_at: Optional[datetime] = None


class AuditInfo(BaseModel):
    """Audit information for tracking data changes."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None


class StatusInfo(BaseModel):
    """Status information for tracking processing state."""
    status: str = "pending"
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None


class Metadata(BaseModel):
    """Generic metadata container."""
    tags: List[str] = []
    properties: Dict[str, Any] = {}
    confidence_score: Optional[float] = None