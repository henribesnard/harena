from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AgentConfig:
    """Configuration for financial agents."""

    name: str
    system_message: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: int = 10
    few_shot_examples: Optional[List[Dict[str, str]]] = None


@dataclass
class AgentResponse:
    """Standard response object returned by agents."""

    agent_name: str
    success: bool
    result: Optional[Dict[str, Any]]
    processing_time_ms: int
    tokens_used: int = 0
    cached: bool = False
    error_message: Optional[str] = None
