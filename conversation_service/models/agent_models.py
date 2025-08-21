from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator, ConfigDict
from datetime import datetime

class AgentResponse(BaseModel):
    """
    Standardized response format from AutoGen agents.
    
    Provides consistent structure for all agent responses with
    metadata, timing, and error handling information.
    """
    
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={
            "example": {
                "agent_name": "intent_classifier",
                "success": True,
                "result": {
                    "intent": "BALANCE_INQUIRY",
                    "confidence": 0.95,
                    "reasoning": "Clear balance request"
                },
                "processing_time_ms": 245,
                "tokens_used": 125,
                "cached": False
            }
        }
    )
    
    agent_name: str = Field(..., description="Name of the agent that generated response")
    success: bool = Field(..., description="Whether the agent processing succeeded")
    
    # Result data (None if failed)
    result: Optional[Dict[str, Any]] = Field(
        None, 
        description="Agent processing result data"
    )
    error_message: Optional[str] = Field(
        None, 
        description="Error message if processing failed"
    )
    
    # Performance metadata
    processing_time_ms: int = Field(
        ..., 
        ge=0, 
        description="Processing time in milliseconds"
    )
    tokens_used: Optional[int] = Field(
        None, 
        ge=0, 
        description="Number of LLM tokens consumed"
    )
    cached: bool = Field(
        default=False, 
        description="Whether result was served from cache"
    )
    
    # Additional metadata
    timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="Response generation timestamp"
    )
    model_used: Optional[str] = Field(
        None, 
        description="LLM model used for processing"
    )
    
    @model_validator(mode='after')
    def validate_success_consistency(self):
        """Validate consistency between success flag and result/error."""
        if self.success and self.result is None:
            raise ValueError("Successful response must include result data")
        
        if not self.success and not self.error_message:
            raise ValueError("Failed response must include error message")
        
        if not self.success and self.result is not None:
            # Allow partial results in failed responses for debugging
            pass
        
        return self
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics for monitoring."""
        return {
            "agent_name": self.agent_name,
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used or 0,
            "cached": self.cached,
            "success": self.success,
            "timestamp": self.timestamp.isoformat()
        }


__all__ = ["AgentResponse"]
