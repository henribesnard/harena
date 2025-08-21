"""Intent classification agent."""

from typing import Any, Dict, Optional

import asyncio

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
        """Classify user intent using few-shot prompting.

        The method builds a chat completion request using the system prompt and
        few-shot examples defined in :mod:`conversation_service.prompts`.  It
        retries the OpenAI call on transient network or timeout errors.
        """

        user_message = input_data.get("user_message", "")
        context = input_data.get("context", {})

        prompt = user_message

        # Retry locally in addition to the retry logic of ``OpenAIClient``
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
                intent = response["content"].strip()
                return {"input": input_data, "context": context, "intent": intent}
            except Exception as exc:  # pragma: no cover - network/timeout
                last_error = exc
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        # Should not reach here; propagate last error for caller
        if last_error:
            raise last_error
        return None
