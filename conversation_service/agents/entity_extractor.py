"""Entity extraction agent."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts.entity_prompts import load_prompt, get_examples
from ..prompts import entity_prompts


class EntityExtractorAgent(BaseFinancialAgent):
    """Extract structured entities from user queries."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="entity_extractor",
            system_message=entity_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = entity_prompts.get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a placeholder list of entities."""
        context = input_data.get("context", {})
        return {"input": input_data, "context": context, "entities": []}
