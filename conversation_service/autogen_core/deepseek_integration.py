"""Intégration DeepSeek dédiée à AutoGen"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from conversation_service.clients.deepseek_client import DeepSeekClient

logger = logging.getLogger("conversation_service.autogen.deepseek")


class DeepSeekAutoGenClient:
    """Wrapper fournissant la configuration LLM pour AutoGen."""

    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient()

    async def initialize(self) -> None:
        await self.client.initialize()

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        return await self.client.chat_completion(messages=messages, **kwargs)

    def get_llm_config(self) -> Dict[str, Any]:
        """Configuration LLM compatible AutoGen."""
        return {
            "config_list": [
                {
                    "model": self.client.model_chat,
                    "api_key": self.client.api_key,
                    "base_url": self.client.api_url,
                }
            ]
        }
