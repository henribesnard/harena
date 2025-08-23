from __future__ import annotations

"""Common utilities for OpenAI-based agents with Pydantic validation."""

import json
from typing import Any, Dict, List, Type

from pydantic import BaseModel

from conversation_service.models.agent_models import AgentConfig


class BaseAgent:
    """Base class for agents relying on the OpenAI client."""

    def __init__(self, client: Any, config: AgentConfig) -> None:
        self.client = client
        self.config = config

    async def _run(
        self, messages: List[Dict[str, str]], response_model: Type[BaseModel]
    ) -> BaseModel:
        """Execute a prompt and parse the response as ``response_model``."""
        response = await self.client.chat_completion(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                *messages,
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            },
            agent_name=self.config.name,
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Invalid response from agent {self.config.name}: {content}"
            ) from exc
