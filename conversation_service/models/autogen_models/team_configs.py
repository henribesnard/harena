"""Modèles Pydantic pour la configuration des équipes d'agents AutoGen"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from .agent_configs import AgentConfig


class TeamConfig(BaseModel):
    """Configuration d'une équipe d'agents"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Nom de l'équipe")
    agents: List[AgentConfig] = Field(..., description="Agents participants")
    max_rounds: int = Field(5, ge=1, description="Nombre maximal d'itérations")
    description: Optional[str] = Field(default=None, description="Description de l'équipe")
    shared_state: bool = Field(False, description="État partagé entre agents")
