"""Configuration des agents Autogen utilisés par le service.

Chaque agent est associé à un modèle OpenAI.  Cette configuration est
séparée de :mod:`openai_config` afin de pouvoir adapter facilement les
modèles utilisés par les agents sans modifier la configuration globale
OpenAI.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration d'un agent individuel."""

    model: str = Field(..., description="Nom du modèle OpenAI")
    temperature: float = Field(0.7, ge=0, le=2, description="Température du modèle")
    max_tokens: int = Field(512, ge=1, description="Nombre maximal de tokens")


class AutoGenConfig(BaseModel):
    """Configuration complète pour tous les agents Autogen."""

    agents: Dict[str, AgentConfig] = Field(default_factory=dict)


DEFAULT_AUTOGEN_CONFIG = AutoGenConfig(
    agents={
        "assistant": AgentConfig(model="gpt-4o-mini"),
        "reasoner": AgentConfig(model="gpt-4o", temperature=0.2, max_tokens=1024),
    }
)
