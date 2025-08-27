"""Runtime de conversation utilisant AutoGen"""
from __future__ import annotations

import logging
from typing import Any, Dict

from .agent_factory import AutoGenAgentFactory
from .conversation_state import ConversationState

logger = logging.getLogger("conversation_service.autogen.runtime")


class ConversationServiceRuntime:
    """Gère les équipes AutoGen et le flux de conversation."""

    def __init__(self, factory: AutoGenAgentFactory | None = None) -> None:
        self.factory = factory or AutoGenAgentFactory()
        self.assistant = None
        self.user_proxy = None

    async def initialize_team(self, system_message: str) -> None:
        """Initialise les agents principaux."""
        self.user_proxy = await self.factory.create_user_proxy("user")
        self.assistant = await self.factory.create_assistant("assistant", system_message)
        logger.info("Equipe AutoGen initialisée")

    async def process_conversation(self, state: ConversationState, user_message: str) -> str:
        """Traite une requête utilisateur et retourne la réponse normalisée."""
        if not self.assistant or not self.user_proxy:
            await self.initialize_team(system_message="")

        state.add_user_message(user_message)
        messages = state.to_messages()
        raw_response = await self.factory.deepseek.chat(messages)
        content = self._normalize_response(raw_response)
        state.add_agent_message(content)
        return content

    def _normalize_response(self, response: Dict[str, Any]) -> str:
        """Extrait proprement le texte de réponse du modèle."""
        try:
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            return str(response).strip()
