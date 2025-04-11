"""
Recurring transaction data models.

This module defines the data structures for recurring transactions,
such as subscriptions and regular bill payments.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID, uuid4
from enum import Enum

from .schema import AuditInfo


class RecurrenceFrequency(str, Enum):
    """Enumeration of recurrence frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"
    IRREGULAR = "irregular"


class RecurrenceStatus(str, Enum):
    """Enumeration of recurrence statuses."""
    ACTIVE = "active"
    ENDED = "ended"
    PAUSED = "paused"
    UNCERTAIN = "uncertain"


class RecurringPatternBase(BaseModel):
    """Base recurring pattern data."""
    frequency: RecurrenceFrequency
    typical_amount: float
    amount_variance: float = 0.0
    typical_day: Optional[int] = None  # Day of month or week
    day_variance: int = 0
    description_pattern: str
    merchant_id: Optional[UUID] = None
    category_id: Optional[int] = None


class RecurringPattern(RecurringPatternBase, AuditInfo):
    """Complete recurring pattern model with internal data."""
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    confidence_score: float = 0.0
    transaction_ids: List[UUID] = []
    first_occurrence: Optional[date] = None
    last_occurrence: Optional[date] = None
    expected_next_date: Optional[date] = None
    status: RecurrenceStatus = RecurrenceStatus.ACTIVE
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        orm_mode = True


class RecurringTransactionBase(BaseModel):
    """Base recurring transaction data."""
    pattern_id: UUID
    transaction_id: UUID
    variance_from_typical: float = 0.0
    days_from_expected: int = 0


class RecurringTransaction(RecurringTransactionBase, AuditInfo):
    """Complete recurring transaction model with internal data."""
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    matched_confidence: float = 1.0
    
    class Config:
        orm_mode = True


class RecurringPatternRead(BaseModel):
    """Public recurring pattern model returned from API endpoints."""
    id: UUID
    frequency: str
    typical_amount: float
    description_pattern: str
    merchant_name: Optional[str] = None
    merchant_logo_url: Optional[str] = None
    category_name: Optional[str] = None
    first_occurrence: Optional[date]
    last_occurrence: Optional[date]
    expected_next_date: Optional[date]
    status: str
    transaction_count: int = 0

    class Config:
        orm_mode = True


class RecurringPatternDetail(RecurringPatternRead):
    """Detailed recurring pattern model with additional information."""
    amount_variance: float
    typical_day: Optional[int]
    day_variance: int
    confidence_score: float
    merchant_id: Optional[UUID]
    category_id: Optional[int]
    created_at: datetime
    updated_at: datetime


class RecurringTransactionRead(BaseModel):
    """Public recurring transaction model returned from API endpoints."""
    id: UUID
    pattern_id: UUID
    transaction_id: UUID
    variance_from_typical: float
    days_from_expected: int
    matched_confidence: float
    created_at: datetime

    class Config:
        orm_mode = True


class RecurringPreview(BaseModel):
    """Preview of detected recurring patterns."""
    patterns: List[RecurringPatternRead]
    total_monthly_amount: float
    total_annual_amount: float
    by_category: Dict[str, float]
    by_frequency: Dict[str, int]