"""Intent classification agent."""

from typing import Any, Dict, Optional

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import intent_prompts


class IntentClassifierAgent(BaseFinancialAgent):
    """Classify user intent using language model prompts."""

    def __init__(self, openai_client):
        config = AgentConfig(
            name="intent_classifier",
            system_message=intent_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = intent_prompts.get_examples()

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a placeholder intent classification result."""
        return {"input": input_data, "intent": "UNKNOWN"}
