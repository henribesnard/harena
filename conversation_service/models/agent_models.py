from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AgentConfig:
    """Configuration container for conversation agents."""

    name: str
    system_message: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 512
    timeout_seconds: int = 30
    few_shot_examples: Optional[List[Dict[str, str]]] = None


__all__ = ["AgentConfig"]
