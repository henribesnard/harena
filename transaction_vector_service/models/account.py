"""
Account data models.

This module defines the data structures for bank accounts
that are associated with transactions.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum

from .schema import AuditInfo


class AccountType(str, Enum):
    """Enumeration of account types."""
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"
    LOAN = "loan"
    INVESTMENT = "investment"
    PENSION = "pension"
    OTHER = "other"


class AccountBase(BaseModel):
    """Base account data shared between creation and reading."""
    bridge_account_id: int
    bridge_item_id: int
    user_id: int
    name: str
    account_type: AccountType
    currency_code: str = "EUR"
    balance: float = 0.0
    is_active: bool = True


class Account(AccountBase, AuditInfo):
    """Complete account model with internal data."""
    id: int
    last_sync_date: Optional[datetime] = None
    transaction_count: int = 0
    iban: Optional[str] = None
    nickname: Optional[str] = None
    color: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        orm_mode = True


class AccountRead(BaseModel):
    """Public account model returned from API endpoints."""
    id: int
    bridge_account_id: int
    name: str
    nickname: Optional[str]
    account_type: str
    currency_code: str
    balance: float
    is_active: bool
    color: Optional[str]
    transaction_count: int
    last_sync_date: Optional[datetime]

    class Config:
        orm_mode = True


class AccountDetail(AccountRead):
    """Detailed account model with additional information."""
    bridge_item_id: int
    iban: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = {}


class AccountStats(BaseModel):
    """Statistical summary of an account."""
    account_id: int
    account_name: str
    current_balance: float
    average_balance: Optional[float] = None
    lowest_balance: Optional[float] = None
    highest_balance: Optional[float] = None
    total_inflow: float = 0
    total_outflow: float = 0
    net_flow: float = 0
    monthly_average_inflow: Optional[float] = None
    monthly_average_outflow: Optional[float] = None
    balance_history: Dict[str, float] = {}  # Date string -> Balance