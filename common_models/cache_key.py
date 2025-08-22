from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, ConfigDict


class DynamicCacheKey(BaseModel):
    """Représente une clé de cache composée dynamiquement.

    La clé peut être fournie sous forme de chaîne, de liste de segments ou
    d'objet dictionnaire.  La méthode :meth:`render` permet de produire une
    chaîne unique exploitable par les backends de cache.
    """

    key: Union[str, List[Any], Dict[str, Any]] = Field(
        ..., description="Structure de la clé (str, liste ou dict)."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées additionnelles associées à la clé.",
    )

    def render(self) -> str:
        """Convertit la clé en chaîne unique."""
        if isinstance(self.key, str):
            return self.key
        if isinstance(self.key, list):
            return ":".join(str(part) for part in self.key)
        return json.dumps(self.key, sort_keys=True)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "key": ["search", 1, {"q": "pizza"}],
                "metadata": {"ttl": 60},
            }
        }
    )
