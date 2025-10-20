"""Response models for conversation endpoints."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TokenUsage(BaseModel):
    """Token usage information."""

    input: int = Field(..., description="Input tokens used")
    output: int = Field(..., description="Output tokens used")
    total: int = Field(..., description="Total tokens used")


class ResponseMetadata(BaseModel):
    """Metadata about the response generation."""

    execution_time_ms: int = Field(..., description="Total execution time in milliseconds")
    tokens_used: TokenUsage = Field(..., description="Token usage breakdown")
    cost_usd: float = Field(..., description="Estimated cost in USD")
    sql_query: str = Field(..., description="Generated SQL query")
    total_transactions_found: int = Field(..., description="Total number of transactions found")
    cached: bool = Field(..., description="Whether the result was cached")
    model_used: str = Field(default="deepseek-chat", description="LLM model used")


class Visualization(BaseModel):
    """Visualization data for charts."""

    type: str = Field(
        ...,
        description="Type of visualization",
        examples=["bar_chart", "line_chart", "pie_chart"]
    )
    title: str = Field(..., description="Chart title")
    data: Dict[str, Any] = Field(..., description="Chart data")


class Insight(BaseModel):
    """Single insight generated from the data."""

    text: str = Field(..., description="Insight text")
    type: Optional[str] = Field(
        None,
        description="Type of insight",
        examples=["trend", "anomaly", "budget_alert"]
    )


class Recommendation(BaseModel):
    """Single recommendation for the user."""

    text: str = Field(..., description="Recommendation text")
    impact: Optional[str] = Field(
        None,
        description="Expected impact",
        examples=["high", "medium", "low"]
    )


class ConversationResponse(BaseModel):
    """Complete response for a conversation request."""

    conversation_id: str = Field(
        ...,
        description="UUID of the conversation"
    )
    user_id: int = Field(
        ...,
        description="ID of the user"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    answer: str = Field(
        ...,
        description="Natural language answer to the user's question"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="List of insights generated from the data"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of recommendations for the user"
    )
    visualization: Optional[Visualization] = Field(
        None,
        description="Visualization data if applicable"
    )
    metadata: ResponseMetadata = Field(
        ...,
        description="Metadata about the response generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": 12345,
                "timestamp": "2025-10-19T14:32:15.123Z",
                "answer": "Ce mois-ci, vous avez dépensé 847,32 € en restaurants répartis sur 127 transactions, soit une moyenne de 6,67 € par repas.",
                "insights": [
                    "Augmentation de 23% par rapport au mois dernier (689€)",
                    "Vous êtes 97€ au-dessus de votre budget restaurants (750€/mois)"
                ],
                "recommendations": [
                    "Réduire les fast-foods de 2-3 fois par semaine pourrait économiser ~60€/mois"
                ],
                "visualization": {
                    "type": "bar_chart",
                    "title": "Dépenses restaurants ce mois vs mois dernier",
                    "data": {
                        "labels": ["Mois dernier", "Ce mois-ci", "Budget"],
                        "values": [689.50, 847.32, 750.00]
                    }
                },
                "metadata": {
                    "execution_time_ms": 987,
                    "tokens_used": {
                        "input": 2150,
                        "output": 285,
                        "total": 2435
                    },
                    "cost_usd": 0.0012,
                    "sql_query": "WITH summary AS (...) SELECT ...",
                    "total_transactions_found": 127,
                    "cached": False,
                    "model_used": "deepseek-chat"
                }
            }
        }


class ErrorDetail(BaseModel):
    """Error details."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    path: str = Field(..., description="Request path")

    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Token JWT invalide ou expiré",
                    "details": "Veuillez vous reconnecter"
                },
                "timestamp": "2025-10-19T14:32:15.123Z",
                "path": "/api/v2/conversation/12345"
            }
        }
