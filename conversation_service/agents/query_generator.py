"""Query generation agent."""

from typing import Any, Dict, Optional

from pydantic import ValidationError
import asyncio
import json

from .query_generator_agent import QueryOptimizer

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import query_prompts
from ..clients import SearchClient
from search_service.models import SearchRequest


class QueryGeneratorAgent(BaseFinancialAgent):
    """Generate search queries based on extracted data."""

    def __init__(self, openai_client, search_client: SearchClient):
        config = AgentConfig(
            name="query_generator",
            system_message=query_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.examples = query_prompts.get_examples()
        self.search_client = search_client

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build and execute a search request based on ``input_data``."""

        context = input_data.get("context", {})
        payload = {
            "user_id": context.get("user_id"),
            "query": context.get("query", ""),
            "filters": context.get("filters", {}),
            "aggregations": context.get("aggregations"),
        }

        try:
            request_model = SearchRequest(**payload)
        except ValidationError:
            # If validation fails we do not query the search service
            return None

        response = await self.search_client.search(request_model.model_dump())

        return {
            "input": input_data,
            "context": context,
            "search_request": request_model.model_dump(),
            "search_response": response,
        }
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
