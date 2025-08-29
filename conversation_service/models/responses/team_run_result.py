"""Model for standardized team execution results."""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict, field_validator

if TYPE_CHECKING:  # pragma: no cover - hints uniquement
    from conversation_service.models.responses.conversation_responses import (
        IntentClassificationResult,
    )
    from conversation_service.models.conversation.entities import (
        EntityExtractionResult,
    )


class TeamRunResult(BaseModel):
    """Résultat d'exécution d'une équipe d'agents."""

    workflow_stage: str = Field(..., description="Étape atteinte du workflow")
    intent_result: Optional[IntentClassificationResult] = Field(
        default=None, description="Résultat de classification d'intention"
    )
    entities_result: Optional[EntityExtractionResult] = Field(
        default=None, description="Résultat d'extraction d'entités"
    )
    workflow_success: bool = Field(
        ..., description="Succès global du workflow"
    )
    coherence_validation: bool = Field(
        ..., description="Validation de cohérence intention/entités"
    )
    agents_sequence: List[str] = Field(
        default_factory=list, description="Ordre d'exécution des agents"
    )
    processing_time_ms: float = Field(
        ..., ge=0, description="Temps de traitement en millisecondes"
    )

    @field_validator("workflow_stage")
    @classmethod
    def validate_workflow_stage(cls, v: str) -> str:
        valid_stages = [
            "intent_classification",
            "entity_extraction",
            "completed",
        ]
        if v not in valid_stages:
            raise ValueError(
                f"workflow_stage must be one of: {valid_stages}"
            )
        return v

    model_config = ConfigDict(validate_assignment=True, extra="forbid")


__all__ = ["TeamRunResult"]
