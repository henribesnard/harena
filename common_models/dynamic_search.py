from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict

from .cache_key import DynamicCacheKey


class DynamicSearchServiceQuery(BaseModel):
    """Requête flexible pour le service de recherche.

    Permet d'envoyer des paramètres arbitraires tout en imposant la présence
    d'un ``user_id`` pour raisons de sécurité et de traçabilité.
    """

    user_id: int = Field(
        ..., description="Identifiant utilisateur (obligatoire)", gt=0
    )
    query: str = Field("", description="Requête textuelle libre")
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Filtres additionnels optionnels"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Métadonnées optionnelles"
    )
    cache_key: Optional[DynamicCacheKey] = Field(
        None, description="Clé de cache éventuellement fournie"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": 42,
                "query": "cafés à Paris",
                "filters": {"category": "food"},
                "metadata": {"debug": True},
                "cache_key": {
                    "key": ["search", 42, "cafes"],
                    "metadata": {"ttl": 30},
                },
            }
        }
    )
