from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class SearchType(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class OperationType(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"
    ALL = "all"

class TimeUnit(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

class DateRange(BaseModel):
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    relative: Optional[str] = None  # "last_week", "last_month", etc.

class AmountRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None

class FilterSet(BaseModel):
    date_range: Optional[DateRange] = None
    amount_range: Optional[AmountRange] = None
    categories: Optional[List[str]] = None
    merchants: Optional[List[str]] = None
    operation_types: Optional[List[OperationType]] = None
    custom_filters: Optional[Dict[str, Any]] = None

class AggregationType(str, Enum):
    SUM = "sum"
    AVG = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    RATIO = "ratio"

class GroupBy(str, Enum):
    NONE = "none"
    CATEGORY = "category"
    MERCHANT = "merchant"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

class AggregationRequest(BaseModel):
    type: AggregationType
    group_by: Optional[GroupBy] = None
    field: str = "amount"
    time_unit: Optional[TimeUnit] = None

class SearchParameters(BaseModel):
    lexical_weight: float = Field(0.5, ge=0.0, le=1.0)
    semantic_weight: float = Field(0.5, ge=0.0, le=1.0)
    top_k_initial: int = Field(50, ge=1, le=1000)
    top_k_final: int = Field(10, ge=1, le=100)

class UserContext(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    previous_questions: Optional[List[str]] = None

class QueryDetails(BaseModel):
    text: str
    expanded_text: Optional[str] = None
    type: Optional[SearchType] = SearchType.HYBRID

class SearchQuery(BaseModel):
    query: QueryDetails
    filters: Optional[FilterSet] = None
    aggregation: Optional[AggregationRequest] = None
    search_params: Optional[SearchParameters] = None
    context: Optional[UserContext] = None