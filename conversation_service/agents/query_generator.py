"""Query generation agent."""

from typing import Any, Dict, Optional

from pydantic import ValidationError

from .query_generator_agent import QueryOptimizer

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts.query_prompts import load_prompt, get_examples
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
