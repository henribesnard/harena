"""Query generation agent."""

from typing import Any, Dict, Optional

from pydantic import ValidationError

from .base_agent import BaseFinancialAgent
from ..models.agent_models import AgentConfig
from ..prompts import query_prompts
from ..clients import SearchClient
from search_service.models import SearchRequest


class QueryGeneratorAgent(BaseFinancialAgent):
    """Generate search queries based on extracted data."""

    def __init__(
        self,
        search_client: SearchClient,
        openai_client: Optional[Any] = None,
    ):
        config = AgentConfig(
            name="query_generator",
            system_message=query_prompts.get_prompt(),
            model_name="gpt-4o-mini",
        )
        super().__init__(config=config, openai_client=openai_client)
        self.search_client = search_client

    async def _process_implementation(
        self, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build and execute a search request based on ``input_data``."""

        if input_data.get("intent") is None or input_data.get("entities") is None:
            return {"error": "intent and entities are required"}

        context = input_data.get("context", {})
        user_id = context.get("user_id")
        filters = dict(context.get("filters", {}))
        filters["user_id"] = user_id

        payload = {
            "user_id": user_id,
            "query": context.get("query", ""),
            "filters": filters,
            "aggregations": context.get("aggregations"),
        }

        try:
            search_request = SearchRequest(**payload)
        except ValidationError:
            # If validation fails we do not query the search service
            return None

        response = await self.search_client.search(
            search_request.user_id, search_request.model_dump()
        )

        return {
            "input": input_data,
            "context": context,
            "search_request": search_request.model_dump(),
            "search_response": response,
        }
