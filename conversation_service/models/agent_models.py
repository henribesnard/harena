"""
Agent models for AutoGen v0.4 integration in Conversation Service MVP.

This module defines the core data models for AutoGen agents, including
configuration, responses, and team workflow management. While initially
optimized for the financial conversation use case, the models now support
OpenAI's GPT family of LLMs.

Classes:
    - AgentConfig: Configuration model for AutoGen agents
    - AgentResponse: Standardized response format from agents
    - TeamWorkflow: Team workflow configuration and management

Author: Conversation Service Team
Created: 2025-01-31
Version: 1.0.0 MVP - Pydantic V2
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime

__all__ = ["AgentConfig", "AgentResponse", "TeamWorkflow"]

# Accepted model name prefixes for OpenAI models supported by the service.
SUPPORTED_MODEL_PREFIXES = (
    "gpt-",
    "o1-",
    "o3-",
    "text-",
    "chatgpt",
)


class AgentConfig(BaseModel):
    """
    Configuration model for AutoGen agents.
    
    This model defines the complete configuration for an AutoGen agent,
    including model client settings, system messages, and behavioral parameters.
    Optimized for financial domain specialization and compatible with multiple
    LLM providers such as OpenAI's GPT models.
    
    Attributes:
        name: Unique identifier for the agent
        model_client_config: LLM model configuration (e.g., OpenAI GPT)
        system_message: System prompt for the agent's behavior
        max_consecutive_auto_reply: Maximum consecutive auto-replies
        description: Optional description of agent's purpose
        agent_type: Type of AutoGen agent (AssistantAgent, UserProxyAgent, etc.)
        temperature: LLM temperature for response generation
        max_tokens: Maximum tokens for response generation
        timeout_seconds: Request timeout in seconds
        retry_attempts: Number of retry attempts on failure
    """
    
    name: str = Field(
        ...,
        description="Unique identifier for the agent",
        min_length=1,
        max_length=100
    )
    
    model_client_config: Dict[str, Any] = Field(
        ...,
        description="LLM model client configuration"
    )
    
    system_message: str = Field(
        ...,
        description="System prompt defining agent behavior",
        min_length=10
    )
    
    max_consecutive_auto_reply: int = Field(
        default=3,
        description="Maximum consecutive auto-replies",
        ge=0,
        le=10
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Optional description of agent purpose",
        max_length=500
    )
    
    agent_type: Literal["AssistantAgent", "UserProxyAgent", "ConversableAgent"] = Field(
        default="AssistantAgent",
        description="Type of AutoGen agent"
    )
    
    temperature: float = Field(
        default=0.1,
        description="LLM temperature for response generation",
        ge=0.0,
        le=2.0
    )
    
    max_tokens: int = Field(
        default=500,
        description="Maximum tokens for response generation",
        gt=0,
        le=4000
    )

    timeout_seconds: int = Field(
        default=20,
        description="Request timeout in seconds",
        gt=0,
        le=120
    )
    
    retry_attempts: int = Field(
        default=2,
        description="Number of retry attempts on failure",
        ge=0,
        le=5
    )
    
    @field_validator("model_client_config")
    @classmethod
    def validate_model_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration for supported LLM providers.

        Currently accepts models from OpenAI's GPT family (``gpt-``) and other
        OpenAI models (e.g. ``o1-`` series).
        """
        required_keys = ["model", "api_key", "base_url"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key '{key}' in model_client_config")

        # Validate model format for known providers using predefined prefixes
        if not any(v["model"].startswith(prefix) for prefix in SUPPORTED_MODEL_PREFIXES):
            raise ValueError(
                "Model must start with a supported prefix: "
                + ", ".join(SUPPORTED_MODEL_PREFIXES)
            )

        return v
    
    @field_validator("system_message")
    @classmethod
    def validate_system_message(cls, v: str) -> str:
        """Validate system message content."""
        if not v.strip():
            raise ValueError("System message cannot be empty")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "intent_classifier_agent",
                "model_client_config": {
                    # Example using an OpenAI GPT model
                    "model": "gpt-4o-mini",
                    "api_key": "sk-xxx",
                    "base_url": "https://api.openai.com/v1"
                },
                "system_message": "You are a financial intent classification agent...",
                "max_consecutive_auto_reply": 3,
                "description": "Classifies user intents in financial conversations",
                "agent_type": "AssistantAgent",
                "temperature": 0.1,
                "max_tokens": 500,
                "timeout_seconds": 30,
                "retry_attempts": 2
            }
        }
    }




class AgentResponse(BaseModel):
    """
    Standardized response format from AutoGen agents.
    
    This model ensures consistent response formatting across all agents
    in the conversation service, including metadata for monitoring and debugging.
    
    Attributes:
        agent_name: Name of the responding agent
        content: Response content from the agent
        metadata: Additional metadata about the response
        execution_time_ms: Execution time in milliseconds
        timestamp: Response generation timestamp
        success: Whether the response was successful
        error_message: Error message if response failed
        token_usage: Token usage statistics
        confidence_score: Confidence score for the response (0.0-1.0)
    """
    
    agent_name: str = Field(
        ...,
        description="Name of the responding agent",
        min_length=1
    )
    
    content: str = Field(
        ...,
        description="Response content from the agent"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    
    execution_time_ms: float = Field(
        ...,
        description="Execution time in milliseconds",
        ge=0.0
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response generation timestamp"
    )
    
    success: bool = Field(
        default=True,
        description="Whether the response was successful"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if response failed"
    )
    
    token_usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage statistics"
    )
    
    confidence_score: Optional[float] = Field(
        default=None,
        description="Confidence score for the response",
        ge=0.0,
        le=1.0
    )

    @model_validator(mode='after')
    def validate_error_consistency(self) -> 'AgentResponse':
        """Validate error message consistency with success flag."""
        if not self.success and not self.error_message:
            raise ValueError("Error message is required when success is False")
        return self

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        },
        "json_schema_extra": {
            "example": {
                "agent_name": "intent_classifier_agent",
                "content": "FINANCIAL_QUERY: transaction search with high confidence",
                "metadata": {
                    "intent_type": "FINANCIAL_QUERY",
                    "confidence": 0.95,
                    "entities_found": 2
                },
                "execution_time_ms": 245.5,
                "success": True,
                "token_usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 25,
                    "total_tokens": 145
                },
                "confidence_score": 0.95
            }
        }
    }


class TeamWorkflow(BaseModel):
    """
    Team workflow configuration for AutoGen agent teams.
    
    This model defines how multiple agents work together in the conversation
    service, including workflow types, agent roles, and execution parameters.
    
    Attributes:
        team_name: Unique identifier for the team
        agents: List of agent names in the team
        workflow_type: Type of workflow execution
        description: Optional description of team purpose
        max_rounds: Maximum conversation rounds
        speaker_selection_method: Method for selecting next speaker
        termination_condition: Condition for terminating workflow
        allow_repeat_speaker: Whether to allow repeat speakers
        priority_order: Priority order for agent selection
    """
    
    team_name: str = Field(
        ...,
        description="Unique identifier for the team",
        min_length=1,
        max_length=100
    )
    
    agents: List[str] = Field(
        ...,
        description="List of agent names in the team",
        min_length=2,
        max_length=10
    )
    
    workflow_type: Literal["sequential", "parallel", "round_robin", "selective"] = Field(
        ...,
        description="Type of workflow execution"
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Optional description of team purpose",
        max_length=500
    )
    
    max_rounds: int = Field(
        default=5,
        description="Maximum conversation rounds",
        gt=0,
        le=20
    )
    
    speaker_selection_method: Literal["auto", "manual", "round_robin", "random"] = Field(
        default="auto",
        description="Method for selecting next speaker"
    )
    
    termination_condition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Condition for terminating workflow"
    )
    
    allow_repeat_speaker: bool = Field(
        default=True,
        description="Whether to allow repeat speakers"
    )
    
    priority_order: Optional[List[str]] = Field(
        default=None,
        description="Priority order for agent selection"
    )

    @field_validator("agents")
    @classmethod
    def validate_agents_unique(cls, v: List[str]) -> List[str]:
        """Validate agent list uniqueness."""
        if len(v) != len(set(v)):
            raise ValueError("Agent names must be unique")
        return v
    
    @model_validator(mode='after')
    def validate_priority_order_consistency(self) -> 'TeamWorkflow':
        """Validate priority order consistency with agents."""
        if self.priority_order is not None:
            if set(self.priority_order) != set(self.agents):
                raise ValueError("Priority order must contain exactly the same agents")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "team_name": "financial_analysis_team",
                "agents": [
                    "orchestrator_agent",
                    "intent_classifier_agent", 
                    "search_query_agent",
                    "response_agent"
                ],
                "workflow_type": "sequential",
                "description": "Team for comprehensive financial query analysis",
                "max_rounds": 5,
                "speaker_selection_method": "auto",
                "allow_repeat_speaker": False,
                "priority_order": [
                    "orchestrator_agent",
                    "intent_classifier_agent",
                    "search_query_agent", 
                    "response_agent"
                ]
            }
        }
    }
