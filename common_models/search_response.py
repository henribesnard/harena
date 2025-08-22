from __future__ import annotations

from typing import Any, Dict, Optional
from http import HTTPStatus

from pydantic import BaseModel, Field, ConfigDict


class SearchServiceResponse(BaseModel):
    """Réponse générique pour le service de recherche."""

    status: HTTPStatus = Field(
        ..., description="Code de statut HTTP de la réponse"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Données retournées par le service"
    )
    error: Optional[str] = Field(
        None, description="Message d'erreur éventuel"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées additionnelles (source, timing, etc.)",
    )

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "status": 200,
                "data": {"results": []},
                "error": None,
                "metadata": {"cache_hit": False},
            }
        },
    )
