from __future__ import annotations

from typing import Any, Dict, Optional, Self

from conversation_service.models.conversation.entities import (
    EntityExtractionResult,
)
from .conversation_responses import ConversationResponse


class AutogenConversationResponse(ConversationResponse):
    """Réponse de conversation enrichie pour les agents autogen"""

    entities: Optional[EntityExtractionResult] = None
    autogen_metadata: Optional[Dict[str, Any]] = None

    def apply_team_results(self, team_results: Dict[str, Any]) -> Self:
        """Applique les résultats d'équipe AutoGen à la réponse."""

        self.autogen_metadata = team_results
        return self
