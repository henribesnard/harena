"""Response generation agent."""

from typing import Any, Dict, Optional

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import response_prompts


class ResponseGeneratorAgent(BaseFinancialAgent):
    """Craft natural language responses from search results."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="response_generator",
            system_message=response_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = response_prompts.get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a placeholder response."""
        return {"input": input_data, "response": ""}
"""Lightweight response generation utilities.

This module provides a minimal asynchronous generator used by the websocket
endpoint to stream back responses. In the real application this would bridge
with the more advanced ``ResponseGeneratorAgent``.
"""

from typing import AsyncGenerator


async def stream_response(message: str) -> AsyncGenerator[str, None]:
    """Yield a simple response for the provided message.

    The implementation is intentionally lightweight to avoid heavy dependencies
    during tests while illustrating how streaming would behave.
    """
    yield f"Response: {message}"
