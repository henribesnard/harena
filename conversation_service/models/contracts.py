from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import json
import logging

logger = logging.getLogger(__name__)

class SearchServiceFilter(BaseModel):
    """
    Filter for Search Service queries based on Elasticsearch structure.
    
    Aligns with the search_service Elasticsearch index structure for
    optimal query performance and accurate filtering.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "field": "category_name.keyword",
                "operator": "term",
                "value": "restaurant",
                "boost": 1.0
            }
        }
    )
    
    field: str = Field(..., description="Elasticsearch field name")
    operator: Literal[
        "term", "terms", "match", "range", "exists", 
        "prefix", "wildcard", "fuzzy"
    ] = Field(..., description="Filter operator type")
    value: Union[str, int, float, List[str], Dict[str, Any]] = Field(
        ..., description="Filter value(s)"
    )
    boost: Optional[float] = Field(None, ge=0.1, le=10.0, description="Query boost factor")
    
    @field_validator('field')
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field exists in Elasticsearch structure."""
        # Based on search_service Elasticsearch structure
        valid_fields = {
            # Text search fields
            "searchable_text", "primary_description", "merchant_name", "category_name",
            
            # Keyword filter fields  
            "category_name.keyword", "merchant_name.keyword", "transaction_type",
            "currency_code", "operation_type",
            
            # Numeric fields
            "amount", "amount_abs", "user_id",
            
            # Date fields
            "date", "month_year", "weekday",
            
            # Boolean fields
            "is_future", "is_deleted"
        }
        
        if v not in valid_fields:
            # Allow dynamic fields but log warning
            logger.warning(f"Using non-standard Elasticsearch field: {v}")
        
        return v
    
    @model_validator(mode='after')
    def validate_operator_value_consistency(self):
        """Validate operator and value consistency."""
        if self.operator == "range":
            if not isinstance(self.value, dict):
                raise ValueError("Range operator requires dict value with gte/lte/gt/lt")
            
            required_keys = {"gte", "lte", "gt", "lt"}
            if not any(key in self.value for key in required_keys):
                raise ValueError("Range value must contain at least one of: gte, lte, gt, lt")
        
        elif self.operator == "terms":
            if not isinstance(self.value, list):
                raise ValueError("Terms operator requires list value")
        
        elif self.operator in ["term", "match", "prefix", "wildcard"]:
            if isinstance(self.value, (list, dict)):
                raise ValueError(f"{self.operator} operator requires simple value")
        
        return self

class SearchServiceQuery(BaseModel):
    """
    Complete query contract for Search Service integration.
    
    Provides structured interface for conversation_service to search_service
    communication with proper validation and optimization hints.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "user_id": 12345,
                "query_text": "restaurants paris",
                "filters": [
                    {
                        "field": "category_name.keyword", 
                        "operator": "term", 
                        "value": "restaurant"
                    }
                ],
                "query_type": "lexical_search",
                "limit": 50,
                "offset": 0,
                "sort_by": "date",
                "sort_order": "desc"
            }
        }
    )
    
    # Required fields
    user_id: int = Field(..., description="User ID for security isolation")
    
    # Query parameters
    query_text: Optional[str] = Field(
        None, 
        max_length=500, 
        description="Free text search query"
    )
    filters: List[SearchServiceFilter] = Field(
        default_factory=list, 
        max_length=20, 
        description="Additional filters"
    )
    
    # Query type and performance
    query_type: Literal[
        "lexical_search", "semantic_search", "hybrid_search", 
        "aggregation_only", "filter_only"
    ] = Field(default="lexical_search", description="Type of search to perform")
    
    # Pagination and sorting
    limit: int = Field(default=50, ge=1, le=500, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")
    sort_by: Optional[str] = Field(
        None, 
        description="Field to sort by (date, amount, relevance)"
    )
    sort_order: Literal["asc", "desc"] = Field(
        default="desc", 
        description="Sort order"
    )
    
    # Advanced options
    include_aggregations: bool = Field(
        default=False, 
        description="Include aggregation results"
    )
    aggregation_types: Optional[List[str]] = Field(
        None, 
        description="Types of aggregations to compute"
    )
    timeout_ms: int = Field(
        default=3000, 
        ge=100, 
        le=10000, 
        description="Query timeout in milliseconds"
    )
    
    # Debugging and optimization
    explain_query: bool = Field(
        default=False, 
        description="Include query explanation in response"
    )
    boost_recent: bool = Field(
        default=True, 
        description="Boost recent transactions in scoring"
    )
    
    @model_validator(mode='after')
    def validate_query_completeness(self):
        """Validate query has sufficient parameters."""
        if not self.query_text and not self.filters:
            raise ValueError("Query must have either query_text or filters")
        
        if self.include_aggregations and not self.aggregation_types:
            # Provide default aggregations
            self.aggregation_types = ["category_breakdown", "monthly_summary"]
        
        return self
    
    def to_elasticsearch_query(self) -> Dict[str, Any]:
        """Convert to Elasticsearch query format."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"user_id": self.user_id}}  # Security isolation
                    ]
                }
            },
            "size": self.limit,
            "from": self.offset,
            "_source": [
                "searchable_text", "primary_description", "merchant_name",
                "amount", "amount_abs", "currency_code", "date", 
                "category_name", "transaction_type"
            ]
        }
        
        # Add text query if provided
        if self.query_text:
            text_query = {
                "multi_match": {
                    "query": self.query_text,
                    "fields": [
                        "searchable_text^2.0",
                        "primary_description^1.5", 
                        "merchant_name^1.8",
                        "category_name^1.0"
                    ],
                    "fuzziness": "AUTO",
                    "type": "best_fields"
                }
            }
            query["query"]["bool"]["must"].append(text_query)
        
        # Add filters
        if self.filters:
            filter_clauses = []
            for filter_item in self.filters:
                if filter_item.operator == "term":
                    filter_clauses.append({
                        "term": {filter_item.field: filter_item.value}
                    })
                elif filter_item.operator == "range":
                    filter_clauses.append({
                        "range": {filter_item.field: filter_item.value}
                    })
                elif filter_item.operator == "match":
                    clause = {
                        "match": {filter_item.field: {"query": filter_item.value}}
                    }
                    if filter_item.boost:
                        clause["match"][filter_item.field]["boost"] = filter_item.boost
                    filter_clauses.append(clause)
            
            if filter_clauses:
                query["query"]["bool"]["filter"] = filter_clauses
        
        # Add sorting
        if self.sort_by:
            query["sort"] = [{self.sort_by: {"order": self.sort_order}}]
        elif self.boost_recent:
            # Default: sort by date descending for recent boost
            query["sort"] = [{"date": {"order": "desc"}}]
        
        # Add aggregations if requested
        if self.include_aggregations and self.aggregation_types:
            aggs = {}
            for agg_type in self.aggregation_types:
                if agg_type == "category_breakdown":
                    aggs["category_breakdown"] = {
                        "terms": {
                            "field": "category_name.keyword",
                            "size": 20
                        },
                        "aggs": {
                            "total_amount": {"sum": {"field": "amount_abs"}},
                            "avg_amount": {"avg": {"field": "amount_abs"}}
                        }
                    }
                elif agg_type == "monthly_summary":
                    aggs["monthly_summary"] = {
                        "date_histogram": {
                            "field": "date",
                            "calendar_interval": "month"
                        },
                        "aggs": {
                            "total_spent": {"sum": {"field": "amount_abs"}},
                            "transaction_count": {"value_count": {"field": "amount"}}
                        }
                    }
            
            if aggs:
                query["aggs"] = aggs
        
        return query

class SearchServiceResponse(BaseModel):
    """
    Structured response from Search Service.
    
    Standardized format for search results with metadata,
    aggregations, and performance information.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "success": True,
                "results": [
                    {
                        "transaction_id": "tx_123456",
                        "searchable_text": "AMAZON.FR Paris",
                        "amount": -45.99,
                        "date": "2024-01-15",
                        "category_name": "shopping"
                    }
                ],
                "total_hits": 156,
                "returned_hits": 50,
                "processing_time_ms": 45,
                "query_id": "q_987654321"
            }
        }
    )
    
    # Response status
    success: bool = Field(..., description="Whether search was successful")
    error_message: Optional[str] = Field(
        None, 
        description="Error message if search failed"
    )
    
    # Search results
    results: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of matching transactions"
    )
    
    # Result metadata
    total_hits: int = Field(
        default=0, 
        ge=0, 
        description="Total number of matching documents"
    )
    returned_hits: int = Field(
        default=0, 
        ge=0, 
        description="Number of documents returned"
    )
    max_score: Optional[float] = Field(
        None, 
        description="Highest relevance score"
    )
    
    # Aggregation results
    aggregations: Optional[Dict[str, Any]] = Field(
        None, 
        description="Aggregation results if requested"
    )
    
    # Performance metadata
    processing_time_ms: int = Field(
        default=0, 
        ge=0, 
        description="Search processing time"
    )
    query_id: str = Field(
        ..., 
        description="Unique query identifier for debugging"
    )
    elasticsearch_took_ms: Optional[int] = Field(
        None, 
        description="Elasticsearch execution time"
    )
    
    # Query debugging (if requested)
    query_explanation: Optional[Dict[str, Any]] = Field(
        None, 
        description="Query explanation for debugging"
    )
    
    @model_validator(mode='after')
    def validate_response_consistency(self):
        """Validate response consistency."""
        if self.success:
            if self.returned_hits != len(self.results):
                raise ValueError("returned_hits must match results length")
            
            if self.returned_hits > self.total_hits:
                raise ValueError("returned_hits cannot exceed total_hits")
        else:
            if not self.error_message:
                raise ValueError("Error message required for failed responses")
        
        return self
    
    def get_category_breakdown(self) -> Optional[Dict[str, Any]]:
        """Extract category breakdown from aggregations."""
        if not self.aggregations:
            return None
        
        category_agg = self.aggregations.get("category_breakdown")
        if not category_agg:
            return None
        
        breakdown = {}
        for bucket in category_agg.get("buckets", []):
            category = bucket.get("key")
            breakdown[category] = {
                "transaction_count": bucket.get("doc_count", 0),
                "total_amount": bucket.get("total_amount", {}).get("value", 0),
                "avg_amount": bucket.get("avg_amount", {}).get("value", 0)
            }
        
        return breakdown
    
    def get_top_merchants(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Extract top merchants from results."""
        merchant_stats = {}
        
        for result in self.results:
            merchant = result.get("merchant_name", "Unknown")
            amount = abs(result.get("amount", 0))
            
            if merchant not in merchant_stats:
                merchant_stats[merchant] = {
                    "merchant_name": merchant,
                    "transaction_count": 0,
                    "total_amount": 0
                }
            
            merchant_stats[merchant]["transaction_count"] += 1
            merchant_stats[merchant]["total_amount"] += amount
        
        # Sort by total amount descending
        top_merchants = sorted(
            merchant_stats.values(),
            key=lambda x: x["total_amount"],
            reverse=True
        )
        
        return top_merchants[:limit]


__all__ = ["SearchServiceFilter", "SearchServiceQuery", "SearchServiceResponse"]
