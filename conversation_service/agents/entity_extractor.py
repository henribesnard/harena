"""Entity extraction agent."""

from typing import Any, Dict, List, Optional

import asyncio
import json

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
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
        """Extract entities from the user message.

        Uses few-shot examples to guide the extraction and retries the OpenAI
        call when transient errors occur.
        """

        user_message = input_data.get("user_message", "")
        context = input_data.get("context", {})

        prompt_parts = [f"Message utilisateur : {user_message}"]
        if context:
            prompt_parts.append(f"Contexte : {json.dumps(context)}")
        prompt = "\n".join(prompt_parts)

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self._call_openai(
                        prompt,
                        few_shot_examples=self.examples,
                    ),
                    timeout=self.config.timeout_seconds,
                )
                raw = response["content"].strip()
                entities: Dict[str, str] = {}
                for part in raw.split(","):
                    if ":" in part:
                        key, value = part.split(":", 1)
                        entities[key.strip()] = value.strip()
                return {"input": input_data, "context": context, "entities": entities}
            except Exception as exc:  # pragma: no cover - network/timeout
                last_error = exc
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        if last_error:
            raise last_error
        return None
