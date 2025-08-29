from typing import Any, Dict

from .conversation_responses import ConversationResponse


class AutogenConversationResponse(ConversationResponse):
    """Réponse de conversation enrichie pour les agents autogen"""

    entities: Dict[str, Any] = {}
    autogen_metadata: Dict[str, Any] = {}
