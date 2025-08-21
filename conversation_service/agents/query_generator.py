"""Query generation agent."""

from typing import Any, Dict, Optional

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts.query_prompts import load_prompt, get_examples
from ..prompts import query_prompts


class QueryGeneratorAgent(BaseFinancialAgent):
    """Generate search queries based on extracted data."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="query_generator",
            system_message=query_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = query_prompts.get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a placeholder search query."""
        context = input_data.get("context", {})
        return {"input": input_data, "context": context, "query": ""}
