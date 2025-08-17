"""
Service contracts for standardized communication between Conversation Service and Search Service.

This module defines the interface contracts that ensure consistent and reliable
communication between the Conversation Service (AutoGen + DeepSeek) and the
Search Service (Elasticsearch). These contracts provide type safety and clear
API specifications.

Classes:
    - QueryMetadata: Metadata for search queries
    - SearchParameters: Configuration for search behavior
    - SearchFilters: Filter specifications for search
    - AggregationRequest: Request for data aggregations
    - SearchServiceQuery: Complete search service request contract
    - ResponseMetadata: Metadata for search responses
    - TransactionResult: Individual transaction result
    - AggregationResult: Results from aggregation requests
    - SearchServiceResponse: Complete search service response contract

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Pydantic V2 FINAL
"""

from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from uuid import uuid4

__all__ = [
    "QueryMetadata",
    "SearchParameters",
    "SearchFilters",
    "AggregationRequest",
    "SearchServiceQuery",
    "ResponseMetadata",
    "TransactionResult",
    "AggregationResult",
    "SearchServiceResponse",
    "validate_search_query_contract",
    "validate_search_response_contract",
    "create_minimal_query",
    "create_error_response",
]


class QueryMetadata(BaseModel):
    """
    Metadata for search queries sent to Search Service.

    This model contains contextual information about the search query,
    including origin, user context, and processing requirements.
    """

    query_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this query",
    )

    conversation_id: str = Field(..., description="ID of the originating conversation")

    user_id: int = Field(..., description="ID of the requesting user", gt=0)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the query was created"
    )

    intent_type: str = Field(
        ..., description="Detected intent that triggered this query", min_length=1
    )

    language: str = Field(
        default="fr",
        description="Language of the original user message",
        pattern=r"^[a-z]{2}$",
    )

    priority: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal", description="Processing priority level"
    )

    timeout_ms: int = Field(
        default=5000, description="Maximum processing time allowed", gt=0, le=30000
    )

    retry_count: int = Field(
        default=0, description="Number of retries attempted", ge=0, le=3
    )

    source_agent: Optional[str] = Field(
        default=None, description="Name of the agent that generated this query"
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: int) -> int:
        """Ensure user_id is provided and positive."""
        if v is None or v <= 0:
            raise ValueError("user_id must be greater than 0")
        return v


class SearchParameters(BaseModel):
    """
    Configuration parameters for search behavior.

    This model defines how the search should be performed, including
    result limits, sorting, and search strategy preferences.
    """

    search_text: Optional[str] = Field(
        default=None,
        description="Text to search for in the Search Service",
    )

    max_results: int = Field(
        default=50, description="Maximum number of results to return", gt=0, le=1000
    )

    offset: int = Field(default=0, description="Number of results to skip", ge=0)

    sort_by: Optional[str] = Field(
        default="relevance", description="Field to sort results by"
    )

    sort_order: Literal["asc", "desc"] = Field(default="desc", description="Sort order")

    include_highlights: bool = Field(
        default=True, description="Whether to include result highlights"
    )

    search_strategy: Literal["exact", "fuzzy", "semantic"] = Field(
        default="semantic",
        description="Strategy for search execution",
    )

    boost_recent: bool = Field(
        default=True, description="Whether to boost recent results"
    )

    include_aggregations: bool = Field(
        default=False, description="Whether to include aggregation data"
    )

    fuzzy_matching: bool = Field(
        default=True, description="Whether to enable fuzzy matching"
    )

    min_score: Optional[float] = Field(
        default=None, description="Minimum relevance score for results", ge=0.0, le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "search_text": "restaurant paris",
                "max_results": 20,
                "offset": 0,
                "sort_by": "date",
                "sort_order": "desc",
                "include_highlights": True,
                "search_strategy": "semantic",
                "boost_recent": True,
                "fuzzy_matching": True,
                "min_score": 0.1,
            }
        }
    }


class SearchFilters(BaseModel):
    """
    Filter specifications for search queries.

    This model defines various filters that can be applied to narrow
    search results based on transaction attributes, dates, amounts, etc.
    """
    
    date: Optional[Dict[str, str]] = Field(
        default=None,
        description="Date filter with 'gte' and 'lte' keys"
    )

    amount: Optional[Dict[str, float]] = Field(
        default=None,
        description="Amount filter with 'gte' and 'lte' keys"
    )

    amount_abs: Optional[Dict[str, float]] = Field(
        default=None,
        description="Absolute amount filter with 'gte' and 'lte' keys"
    )

    category_name: Optional[List[str]] = Field(
        default=None,
        description="List of transaction categories to include",
        alias="categories",  # Backwards compatibility
    )

    merchant_name: Optional[List[str]] = Field(
        default=None,
        description="List of merchants to include",
        alias="merchants",  # Backwards compatibility
    )

    transaction_types: Optional[List[str]] = Field(
        default=None, description="List of transaction types to include"
    )

    operation_type: Optional[str] = Field(
        default=None, description="Operation type to include"
    )

    account_ids: Optional[List[str]] = Field(
        default=None, description="List of account IDs to include"
    )

    currencies: Optional[List[str]] = Field(
        default=None, description="List of currencies to include"
    )

    tags: Optional[List[str]] = Field(
        default=None, description="List of tags to include"
    )

    exclude_categories: Optional[List[str]] = Field(
        default=None, description="Categories to exclude"
    )

    exclude_merchants: Optional[List[str]] = Field(
        default=None, description="Merchants to exclude"
    )

    text_query: Optional[str] = Field(
        default=None,
        description="Free text query for description matching",
        max_length=500,
    )

    custom_filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional custom filters"
    )

    user_id: Optional[int] = Field(
        default=None, description="Identifier of the user executing the query", gt=0
    )

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Validate date filter has proper gte/lte."""
        if v is None:
            return v
        # Ensure both bounds are provided
        if "gte" not in v or "lte" not in v:
            raise ValueError("date filter must contain both 'gte' and 'lte'")

        try:
            gte = datetime.fromisoformat(v["gte"])
            lte = datetime.fromisoformat(v["lte"])
        except ValueError as exc:  # pragma: no cover - pydantic already validates format
            raise ValueError("invalid date format") from exc

        if gte > lte:
            raise ValueError("'gte' must be less than or equal to 'lte'")
        return v

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate amount filter has proper gte/lte.

        Accepts filters containing either ``gte`` or ``lte`` or both. Any
        extraneous keys are ignored in the returned value.
        """
        if v is None:
            return v

        allowed = {"gte", "lte"}
        if not any(key in v for key in allowed):
            raise ValueError("amount filter must contain 'gte' or 'lte'")

        if "gte" in v and "lte" in v and v["gte"] > v["lte"]:
            raise ValueError("'gte' must be less than or equal to 'lte'")

        return {k: v[k] for k in allowed if k in v}

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "date": {
                    "gte": "2024-01-01",
                    "lte": "2024-01-31"
                },
                "amount": {
                    "gte": 100.0,
                    "lte": 1000.0
                },
                "categories": ["food", "transport"],
                "merchants": ["Carrefour", "SNCF"],
                "category_name": ["food", "transport"],
                "merchant_name": ["Carrefour", "SNCF"],
                "transaction_types": ["debit"],
                "operation_type": "card",
                "text_query": "restaurant paris",
            }
        },
    }

class AggregationRequest(BaseModel):
    """
    Request for data aggregations (optional).

    This model defines aggregation requests for analytical queries,
    such as grouping by category, calculating sums, averages, etc.
    """

    group_by: Optional[List[str]] = Field(
        default=None, description="Fields to group aggregations by"
    )

    metrics: Optional[List[str]] = Field(
        default=None, description="List of metrics to calculate (sum, avg, count, etc.)"
    )

    date_histogram: Optional[Dict[str, str]] = Field(
        default=None, description="Date histogram configuration"
    )

    top_values: Optional[Dict[str, Any]] = Field(
        default=None, description="Configuration for top values aggregation"
    )

    custom_aggregations: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom aggregation definitions"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "group_by": ["category"],
                "metrics": ["sum", "count", "avg"],
                "date_histogram": {"field": "date", "interval": "month"},
                "top_values": {"field": "merchant", "size": 10},
            }
        }
    }


class SearchServiceQuery(BaseModel):
    """
    Complete search service request contract.

    This is the main contract for requests sent from Conversation Service
    to Search Service, containing all necessary information for processing.
    """

    query_metadata: QueryMetadata = Field(..., description="Metadata about the query")

    search_parameters: SearchParameters = Field(
        ..., description="Search behavior configuration"
    )

    filters: SearchFilters = Field(..., description="Search filters to apply")

    aggregations: Optional[AggregationRequest] = Field(
        default=None, description="Optional aggregation requests"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query_metadata": {
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440001",
                    "user_id": 12345,
                    "intent_type": "SEARCH_BY_DATE",
                    "source_agent": "search_query_agent",
                },
                "search_parameters": {
                    "max_results": 20,
                    "sort_by": "date",
                    "sort_order": "desc",
                    "search_strategy": "semantic",
                },
                "filters": {
                    "date": {"gte": "2024-01-01", "lte": "2024-01-31"},
                    "amount": {"gte": 100.0, "lte": 1000.0},
                    "category_name": ["food", "transport"],
                },
                "aggregations": {
                    "group_by": ["category_name"],
                    "metrics": ["sum"],
                },
            }
        }
    }

    def to_search_request(self) -> Dict[str, Any]:
        """Convert this query to the simplified SearchRequest schema."""
        filters_dict = (
            self.filters.dict(exclude_none=True) if self.filters else {}
        )
        filters_dict.pop("user_id", None)
        return {
            "user_id": self.query_metadata.user_id,
            "query": self.search_parameters.search_text or "",
            "filters": filters_dict,
            "limit": self.search_parameters.max_results,
            "offset": self.search_parameters.offset,
            "metadata": {
                "conversation_id": self.query_metadata.conversation_id,
                "intent_type": self.query_metadata.intent_type,
                "source_agent": self.query_metadata.source_agent,
            },
        }


class ResponseMetadata(BaseModel):
    """
    Metadata for search service responses.

    This model contains information about the search execution,
    performance metrics, and result statistics.
    """

    query_id: str = Field(..., description="ID of the original query")

    response_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the response was generated"
    )

    processing_time_ms: float = Field(
        ..., description="Time taken to process the query", ge=0.0
    )

    total_results: int = Field(
        ..., description="Total number of matching results", ge=0
    )

    returned_results: int = Field(..., description="Number of results returned", ge=0)

    has_more_results: bool = Field(
        ..., description="Whether more results are available"
    )

    search_strategy_used: str = Field(..., description="Actual search strategy used")

    elasticsearch_took: Optional[int] = Field(
        default=None, description="Time Elasticsearch took (internal)", ge=0
    )

    cache_hit: bool = Field(
        default=False, description="Whether the result was served from cache"
    )

    warnings: Optional[List[str]] = Field(
        default=None, description="Any warnings during processing"
    )

    debug_info: Optional[Dict[str, Any]] = Field(
        default=None, description="Debug information (only in development)"
    )

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class TransactionResult(BaseModel):
    """
    Individual transaction result from search.

    This model represents a single transaction returned by the search service,
    with all relevant transaction data and search-specific metadata.
    """

    transaction_id: str = Field(..., description="Unique transaction identifier")

    date: str = Field(..., description="Transaction date (ISO format)")

    amount: float = Field(..., description="Transaction amount")

    currency: str = Field(
        ...,
        description="Transaction currency",
        pattern=r"^[A-Z]{3}$",
        validation_alias="currency_code",
    )

    description: str = Field(
        ...,
        description="Transaction description",
        validation_alias="primary_description",
    )

    merchant: Optional[str] = Field(
        default=None,
        description="Merchant name",
        validation_alias="merchant_name",
    )

    category: Optional[str] = Field(
        default=None,
        description="Transaction category",
        validation_alias="category_name",
    )

    account_id: str = Field(..., description="Account identifier")

    transaction_type: Literal["debit", "credit"] = Field(
        ..., description="Type of transaction"
    )

    operation_type: Optional[str] = Field(
        default=None,
        description="Type of operation",
        validation_alias="operation_type",
    )

    balance_after: Optional[float] = Field(
        default=None, description="Account balance after transaction"
    )

    tags: Optional[List[str]] = Field(
        default=None, description="List of associated tags"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional transaction metadata"
    )

    relevance_score: Optional[float] = Field(
        default=None, description="Search relevance score", ge=0.0, le=1.0
    )

    highlights: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Highlighted text matches"
    )

    @field_validator("account_id", mode="before")
    @classmethod
    def convert_account_id(cls, v: Any) -> str:
        """Ensure account_id is always stored as a string."""
        if isinstance(v, int):
            return str(v)
        return v

    def to_source(self) -> Dict[str, Any]:
        """Return transaction data using alias names for display."""
        dump = getattr(self, "model_dump", None)
        if callable(dump):
            return dump(by_alias=True)
        return self.dict(by_alias=True)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "transaction_id": "txn_123456789",
                "date": "2024-01-15T10:30:00Z",
                "amount": -45.50,
                "currency": "EUR",
                "description": "CARREFOUR PARIS 15",
                "merchant": "Carrefour",
                "category": "food",
                "account_id": "acc_987654321",
                "transaction_type": "debit",
                "balance_after": 1254.50,
                "tags": ["grocery", "essential"],
                "relevance_score": 0.95,
                "highlights": {
                    "description": ["<em>CARREFOUR</em> PARIS 15"],
                    "merchant": ["<em>Carrefour</em>"],
                },
            }
        },
    }


class AggregationResult(BaseModel):
    """
    Results from aggregation requests (optional).

    This model contains the results of data aggregations requested
    in the search query, such as category summaries, time-based groupings, etc.
    """

    aggregation_type: str = Field(
        ..., description="Type of aggregation performed", min_length=1
    )

    results: Dict[str, Any] = Field(..., description="Aggregation results data")

    total_count: int = Field(..., description="Total count across all buckets", ge=0)

    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional aggregation metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "aggregation_type": "category_summary",
                "results": {
                    "buckets": [
                        {"key": "food", "doc_count": 25, "total_amount": -1250.75},
                        {"key": "transport", "doc_count": 12, "total_amount": -340.20},
                    ]
                },
                "total_count": 37,
                "metadata": {
                    "date": "2024-01-01 to 2024-01-31",
                    "currency": "EUR"
                }
            }
        }
    }


class SearchServiceResponse(BaseModel):
    """
    Complete search service response contract.

    This is the main contract for responses sent from Search Service
    back to Conversation Service, containing search results and metadata.
    """

    response_metadata: ResponseMetadata = Field(
        ..., description="Metadata about the response"
    )

    results: List[TransactionResult] = Field(
        ..., description="List of transaction results"
    )

    aggregations: Optional[List[AggregationResult]] = Field(
        default=None, description="Optional aggregation results"
    )

    success: bool = Field(default=True, description="Whether the search was successful")

    error_message: Optional[str] = Field(
        default=None, description="Error message if search failed"
    )

    suggestions: Optional[List[str]] = Field(
        default=None, description="Suggestions for query improvement"
    )

    @model_validator(mode="after")
    def validate_error_consistency(self) -> "SearchServiceResponse":
        """Validate error message consistency with success flag."""
        if not self.success and not self.error_message:
            raise ValueError("Error message is required when success is False")
        return self

    @model_validator(mode="after")
    def validate_results_consistency(self) -> "SearchServiceResponse":
        """Validate results consistency with metadata."""
        if len(self.results) != self.response_metadata.returned_results:
            raise ValueError(
                "Number of results must match returned_results in metadata"
            )
        return self

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the search results."""
        if not self.results:
            return {
                "total_transactions": 0,
                "total_amount": 0.0,
                "avg_amount": 0.0,
                "date": None,
                "categories": [],
                "merchants": [],
                "category_name": [],
                "merchant_name": [],
        }

        amounts = [r.amount for r in self.results]
        dates = [r.date for r in self.results]
        categories = list(set(r.category for r in self.results if r.category))
        merchants = list(set(r.merchant for r in self.results if r.merchant))

        return {
            "total_transactions": len(self.results),
            "total_amount": sum(amounts),
            "avg_amount": sum(amounts) / len(amounts),
            "date": {
                "start": min(dates),
                "end": max(dates)
            } if dates else None,
            "categories": categories,
            "merchants": merchants,
            "category_name": categories,
            "merchant_name": merchants,
        }

    def filter_by_amount(
        self, min_amount: Optional[float] = None, max_amount: Optional[float] = None
    ) -> List[TransactionResult]:
        """Filter results by amount range."""
        filtered = self.results

        if min_amount is not None:
            filtered = [r for r in filtered if r.amount >= min_amount]

        if max_amount is not None:
            filtered = [r for r in filtered if r.amount <= max_amount]

        return filtered

    def filter_by_operation_type(self, operation_type: Optional[str] = None) -> List[TransactionResult]:
        """Filter results by operation type."""
        if not operation_type:
            return self.results
        return [r for r in self.results if r.operation_type == operation_type]

    def group_by_category(self) -> Dict[str, List[TransactionResult]]:
        """Group results by transaction category."""
        groups = {}
        for result in self.results:
            category = result.category or "uncategorized"
            if category not in groups:
                groups[category] = []
            groups[category].append(result)

        return groups

    model_config = {
        "json_schema_extra": {
            "example": {
                "response_metadata": {
                    "query_id": "550e8400-e29b-41d4-a716-446655440002",
                    "processing_time_ms": 125.5,
                    "total_results": 47,
                    "returned_results": 20,
                    "has_more_results": True,
                    "search_strategy_used": "semantic",
                    "cache_hit": False,
                },
                "results": [
                    {
                        "transaction_id": "txn_123456789",
                        "date": "2024-01-15T10:30:00Z",
                        "amount": -45.50,
                        "currency": "EUR",
                        "description": "CARREFOUR PARIS 15",
                        "merchant": "Carrefour",
                        "category": "food",
                        "account_id": "acc_987654321",
                        "transaction_type": "debit",
                        "relevance_score": 0.95,
                    }
                ],
                "aggregations": [
                    {
                        "aggregation_type": "category_summary",
                        "results": {
                            "buckets": [
                                {
                                    "key": "food",
                                    "doc_count": 25,
                                    "total_amount": -1250.75,
                                }
                            ]
                        },
                        "total_count": 25,
                    }
                ],
                "success": True,
                "suggestions": [
                    "Try broadening your date range for more results",
                    "Consider searching for similar merchants like 'Monoprix'",
                ],
            }
        }
    }


# Utility functions for contract validation and conversion


def validate_search_query_contract(query_dict: Dict[str, Any]) -> SearchServiceQuery:
    """
    Validate and convert dictionary to SearchServiceQuery.

    Args:
        query_dict: Dictionary representation of query

    Returns:
        Validated SearchServiceQuery instance

    Raises:
        ValidationError: If validation fails
    """
    return SearchServiceQuery(**query_dict)


def validate_search_response_contract(
    response_dict: Dict[str, Any],
) -> SearchServiceResponse:
    """
    Validate and convert dictionary to SearchServiceResponse.

    Args:
        response_dict: Dictionary representation of response

    Returns:
        Validated SearchServiceResponse instance

    Raises:
        ValidationError: If validation fails
    """
    return SearchServiceResponse(**response_dict)


def create_minimal_query(
    conversation_id: str, user_id: int, intent_type: str
) -> SearchServiceQuery:
    """
    Create a minimal search query with default parameters.

    Args:
        conversation_id: ID of the conversation
        user_id: ID of the user
        intent_type: Detected intent type

    Returns:
        SearchServiceQuery with minimal configuration
    """
    return SearchServiceQuery(
        query_metadata=QueryMetadata(
            conversation_id=conversation_id, user_id=user_id, intent_type=intent_type
        ),
        search_parameters=SearchParameters(),
        filters=SearchFilters(),
    )


def create_error_response(query_id: str, error_message: str) -> SearchServiceResponse:
    """
    Create an error response for failed searches.

    Args:
        query_id: ID of the original query
        error_message: Description of the error

    Returns:
        SearchServiceResponse indicating failure
    """
    return SearchServiceResponse(
        response_metadata=ResponseMetadata(
            query_id=query_id,
            processing_time_ms=0.0,
            total_results=0,
            returned_results=0,
            has_more_results=False,
            search_strategy_used="none",
        ),
        results=[],
        success=False,
        error_message=error_message,
    )
