"""
Financial insight data models.

This module defines the data structures for financial insights,
which are analyses and recommendations based on transaction data.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from uuid import UUID, uuid4
from enum import Enum

from .schema import AuditInfo


class InsightType(str, Enum):
    """Enumeration of insight types."""
    SPENDING_PATTERN = "spending_pattern"
    BUDGET_ALERT = "budget_alert"
    RECURRING_DETECTION = "recurring_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    MERCHANT_PATTERN = "merchant_pattern"
    SAVING_OPPORTUNITY = "saving_opportunity"
    FINANCIAL_HEALTH = "financial_health"
    CUSTOM_ALERT = "custom_alert"


class InsightSeverity(str, Enum):
    """Enumeration of insight severities."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InsightTimeframe(str, Enum):
    """Enumeration of insight timeframes."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class InsightBase(BaseModel):
    """Base insight data shared between creation and reading."""
    type: InsightType
    title: str
    description: str
    severity: InsightSeverity = InsightSeverity.INFO
    timeframe: InsightTimeframe
    start_date: date
    end_date: Optional[date] = None
    category_id: Optional[int] = None
    merchant_id: Optional[UUID] = None
    amount: Optional[float] = None
    previous_amount: Optional[float] = None
    percent_change: Optional[float] = None


class InsightCreate(InsightBase):
    """Model for creating a new insight."""
    user_id: int
    transaction_ids: List[UUID] = []
    data_points: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Insight(InsightBase, AuditInfo):
    """Complete insight model with internal data."""
    id: UUID = Field(default_factory=uuid4)
    user_id: int
    is_read: bool = False
    read_at: Optional[datetime] = None
    is_dismissed: bool = False
    dismissed_at: Optional[datetime] = None
    is_actionable: bool = False
    action_taken: bool = False
    action_taken_at: Optional[datetime] = None
    confidence_score: float = 1.0
    transaction_ids: List[UUID] = []
    data_points: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        orm_mode = True


class InsightRead(BaseModel):
    """Public insight model returned from API endpoints."""
    id: UUID
    type: str
    title: str
    description: str
    severity: str
    timeframe: str
    start_date: date
    end_date: Optional[date]
    category_name: Optional[str] = None
    merchant_name: Optional[str] = None
    amount: Optional[float]
    previous_amount: Optional[float]
    percent_change: Optional[float]
    is_read: bool
    is_dismissed: bool
    is_actionable: bool
    created_at: datetime

    class Config:
        orm_mode = True


class InsightDetail(InsightRead):
    """Detailed insight model with additional information."""
    category_id: Optional[int]
    merchant_id: Optional[UUID]
    data_points: Dict[str, Any]
    action_taken: bool
    action_taken_at: Optional[datetime]
    confidence_score: float
    read_at: Optional[datetime]
    dismissed_at: Optional[datetime]
    transaction_ids: List[UUID]
    updated_at: datetime


class InsightSummary(BaseModel):
    """Summary of insights for a user."""
    total_insights: int
    unread_insights: int
    actionable_insights: int
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    by_timeframe: Dict[str, int]