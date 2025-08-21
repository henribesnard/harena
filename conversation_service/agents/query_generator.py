"""Query generation agent."""

from typing import Any, Dict, Optional

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from prompts.query_prompts import load_prompt, get_examples


class QueryGeneratorAgent(BaseFinancialAgent):
    """Generate search queries based on extracted data."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="query_generator",
            system_message=load_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a placeholder search query."""
        return {"input": input_data, "query": ""}
