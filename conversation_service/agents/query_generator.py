"""Query generation agent."""

from typing import Any, Dict, Optional

import asyncio
import json

from .query_generator_agent import QueryOptimizer

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
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
        """Generate an Elasticsearch query from the conversation context.

        Combines the user message, detected intent and extracted entities to
        produce a structured query.  OpenAI calls are retried on transient
        failures and the output is parsed as JSON when possible.
        """

        user_message = input_data.get("user_message", "")
        context = input_data.get("context", {})

        ctx_lines = []
        if context:
            for key, value in context.items():
                ctx_lines.append(f"{key.upper()}: {value}")
        ctx_block = "\n".join(ctx_lines)
        if ctx_block:
            prompt = f"Message utilisateur : {user_message}\n{ctx_block}\nGénère une requête Elasticsearch optimisée au format SearchServiceQuery."""
        else:
            prompt = f"Message utilisateur : {user_message}\nGénère une requête Elasticsearch optimisée au format SearchServiceQuery."""

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
                content = response["content"].strip()
                try:
                    query = json.loads(content)
                except Exception:
                    query = content
                return {"input": input_data, "context": context, "query": query}
            except Exception as exc:  # pragma: no cover - network/timeout
                last_error = exc
                if attempt >= 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        if last_error:
            raise last_error
        return None
