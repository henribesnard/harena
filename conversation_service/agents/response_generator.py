"""Response generation agent."""

from typing import Any, Dict, Optional

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from prompts.response_prompts import load_prompt, get_examples


class ResponseGeneratorAgent(BaseFinancialAgent):
    """Craft natural language responses from search results."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="response_generator",
            system_message=load_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a placeholder response."""
        return {"input": input_data, "response": ""}
