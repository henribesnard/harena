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
        if user_id is not None:
            filters.setdefault("user_id", user_id)
        payload = {
            "user_id": user_id,
            "query": context.get("query", ""),
            "filters": filters,
            "aggregations": context.get("aggregations"),
        }

        try:
            request_model = SearchRequest(**payload)
        except ValidationError:
            # If validation fails we do not query the search service
            return None

        response = await self.search_client.search(
            request_model.user_id, request_model.model_dump()
        )

        return {
            "input": input_data,
            "context": context,
            "search_request": request_model.model_dump(),
            "search_response": response,
        }
