"""Modèles Pydantic pour la configuration des agents AutoGen"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class AgentConfig(BaseModel):
    """Configuration d'un agent AutoGen"""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Nom unique de l'agent")
    llm: str = Field(..., description="Modèle LLM utilisé, ex: gpt-4o-mini")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(256, ge=1)
    teachable: bool = Field(False, description="L'agent peut-il apprendre ?")
    memory_enabled: bool = Field(False, description="Mémoire de conversation activée")
    system_prompt: Optional[str] = Field(default=None, description="Prompt système spécifique")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Paramètres additionnels")
