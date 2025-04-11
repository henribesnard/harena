# transaction_vector_service/models/category.py
"""
Category data models.

This module defines the data structures for transaction categories,
which are used to classify and organize financial transactions.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class CategoryLevel(str, Enum):
    """Enumeration of category hierarchy levels."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


class CategoryType(str, Enum):
    """Enumeration of category types."""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    INVESTMENT = "investment"
    LOAN = "loan"
    OTHER = "other"


class CategoryBase(BaseModel):
    """Base category data shared between creation and reading."""
    bridge_category_id: int
    name: str
    display_name: Optional[str] = None
    parent_id: Optional[int] = None
    level: CategoryLevel = CategoryLevel.SECONDARY
    type: CategoryType = CategoryType.EXPENSE
    icon: Optional[str] = None
    color: Optional[str] = None


class CategoryCreate(CategoryBase):
    """Model for creating a new category."""
    keywords: List[str] = []
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Category(CategoryBase):
    """Complete category model with internal data."""
    id: int  # Using Bridge category_id as our primary key
    path: List[int] = []  # List of ancestor category IDs
    keywords: List[str] = []
    transaction_count: int = 0
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True


class CategoryRead(BaseModel):
    """Public category model returned from API endpoints."""
    id: int
    name: str
    display_name: Optional[str]
    parent_id: Optional[int]
    level: str
    type: str
    icon: Optional[str]
    color: Optional[str]
    transaction_count: int

    class Config:
        orm_mode = True


class CategoryDetail(CategoryRead):
    """Detailed category model with additional information."""
    path: List[int]
    keywords: List[str]
    is_active: bool
    children: List[CategoryRead] = []
    created_at: datetime
    updated_at: datetime


class CategoryHierarchy(BaseModel):
    """Model representing the category hierarchy."""
    id: int
    name: str
    display_name: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    type: str
    children: List["CategoryHierarchy"] = []


# This is needed for the self-referencing model
CategoryHierarchy.model_rebuild()

class CategoryStats(BaseModel):
    """Statistical summary of a category."""
    category_id: int
    category_name: str
    transaction_count: int
    total_amount: float
    average_amount: float
    percentage_of_total: float
    by_month: Dict[str, float]  # Month -> Amount