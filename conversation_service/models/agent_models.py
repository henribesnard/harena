"""Pydantic models related to agent configuration and execution."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentStep(BaseModel):
    """Single step executed by an agent."""

    agent: str
    status: str

    @field_validator("agent", "status")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


class AgentTrace(BaseModel):
    """Trace of agent steps with execution time."""

    steps: List[AgentStep] = Field(default_factory=list)
    total_time_ms: float

    @field_validator("total_time_ms")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("total_time_ms must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_steps(self) -> "AgentTrace":
        if not self.steps:
            raise ValueError("steps cannot be empty")
        return self


class AgentConfig(BaseModel):
    """Configuration d'un agent conversationnel."""

    model: str = Field(..., description="Nom du modèle OpenAI")
    temperature: float = Field(0.7, description="Température du modèle")
    max_tokens: int = Field(512, description="Nombre maximal de tokens générés")
    timeout: int = Field(30, description="Délai maximum de génération en secondes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "gpt-4o-mini",
                "temperature": 0.5,
                "max_tokens": 1024,
                "timeout": 30,
            }
        }
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if not 1 <= v < 4000:
            raise ValueError("max_tokens must be between 1 and 3999")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if not 1 <= v <= 60:
            raise ValueError("timeout must be between 1 and 60 seconds")
        return v


class IntentResult(BaseModel):
    """Résultat de la classification d'intention."""

    intent_type: str = Field(..., description="Intention détectée")
    confidence_score: float = Field(..., description="Score de confiance pour l'intention")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent_type": "CHECK_BALANCE",
                "confidence_score": 0.94,
            }
        }
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v


class DynamicFinancialEntity(BaseModel):
    """Entité financière extraite dynamiquement d'un message."""

    entity_type: str = Field(..., description="Type de l'entité")
    value: str = Field(..., description="Valeur associée à l'entité")
    confidence_score: float = Field(..., description="Score de confiance de l'entité")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_type": "account_number",
                "value": "FR7612345678901234567890189",
                "confidence_score": 0.87,
            }
        }
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v


class AgentResponse(BaseModel):
    """Réponse complète retournée par la chaîne d'agents."""

    response: str = Field(..., description="Texte de la réponse générée")
    intent: IntentResult = Field(..., description="Intention détectée")
    entities: List[DynamicFinancialEntity] = Field(
        default_factory=list, description="Entités financières extraites"
    )
    confidence_score: float = Field(
        ..., description="Score global de confiance pour la réponse"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "Votre solde est de 50€.",
                "intent": {
                    "intent_type": "CHECK_BALANCE",
                    "confidence_score": 0.94,
                },
                "entities": [
                    {
                        "entity_type": "account_number",
                        "value": "FR7612345678901234567890189",
                        "confidence_score": 0.87,
                    }
                ],
                "confidence_score": 0.92,
            }
        }
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return v
