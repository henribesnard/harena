from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class ResultType(str, Enum):
    TRANSACTION = "transaction"
    ACCOUNT = "account"
    CATEGORY = "category"
    MERCHANT = "merchant"
    AGGREGATE = "aggregate"

class MatchDetails(BaseModel):
    lexical_score: Optional[float] = None
    semantic_score: Optional[float] = None
    reranking_score: Optional[float] = None

class PaginationInfo(BaseModel):
    page: int
    size: int
    total_pages: int
    next_page: Optional[int] = None
    prev_page: Optional[int] = None

class SearchResult(BaseModel):
    id: str
    type: ResultType
    content: Dict[str, Any]
    score: float
    match_details: Optional[MatchDetails] = None
    highlight: Optional[Dict[str, List[str]]] = None

class ResponseMetadata(BaseModel):
    total_count: int
    filtered_count: int
    returned_count: int
    execution_time_ms: int
    data_freshness: datetime
    search_type: str
    query_parsed: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    aggregations: Optional[Dict[str, Any]] = None
    metadata: ResponseMetadata
    pagination: Optional[PaginationInfo] = None